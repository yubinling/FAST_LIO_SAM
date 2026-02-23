#!/usr/bin/env python
import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 让相对模型路径、config 和 lib_cpp.so 都从检测目录解析，和实时节点保持一致。
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pcl2
import tensorflow as tf

import config.config as cfg
import lib_cpp
from networks.model import livox_model
from realtime_detection_limot import HEIGHT, WIDTH, CHANNELS, OVERLAP, X_MIN, Y_MIN, Y_MAX, Z_MIN, Z_MAX, DZ, DX, DY
from realtime_detection_limot import TrtRunner


def data2voxel(cloud_msg):
    # voxel 构建逻辑要和实时检测节点一致，否则后端对比会混入预处理差异。
    data = [0 for _ in range(HEIGHT * WIDTH * CHANNELS)]
    for point in pcl2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        if x == 0 and y == 0 and z == 0:
            continue
        if np.abs(x) < 2.0 and np.abs(y) < 1.5:
            continue
        if y > Y_MIN and y < Y_MAX and x > X_MIN and x < cfg.RANGE['X_MAX'] and z > Z_MIN and z < Z_MAX:
            channel = int((-z + Z_MAX) / DZ)
            if abs(x) < 3 and abs(y) < 3:
                continue
            if x > -OVERLAP:
                pixel_x = int((x - X_MIN + 2 * OVERLAP) / DX)
                pixel_y = int((-y + Y_MAX) / DY)
                data[pixel_x * WIDTH * CHANNELS + pixel_y * CHANNELS + channel] = 1
            if x < OVERLAP:
                pixel_x = int((-x + OVERLAP) / DX)
                pixel_y = int((y + Y_MAX) / DY)
                data[pixel_x * WIDTH * CHANNELS + pixel_y * CHANNELS + channel] = 1
    return np.reshape(data, (1, HEIGHT, WIDTH, CHANNELS)).astype(np.float32)


class TfRunner(object):
    def __init__(self):
        # 普通网络路径保持 TensorFlow 推理，用来作为 TRT 输出的一致性基准。
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/gpu:' + str(cfg.GPU_INDEX)):
                self.input_bev_img_pl = self.net.placeholder_inputs(cfg.BATCH_SIZE)
                end_points = self.net.get_model(self.input_bev_img_pl)
                self.feature_out = end_points['feature_out']
                saver = tf.train.Saver()
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                config.log_device_placement = False
                self.sess = tf.Session(config=config)
                saver.restore(self.sess, cfg.MODEL_PATH)

    def infer(self, batch_bev_img):
        return self.sess.run(self.feature_out, feed_dict={self.input_bev_img_pl: batch_bev_img})

    def close(self):
        self.sess.close()


def postprocess(feature_out):
    # 两个后端共用 C++ 后处理，最终比较的是 LIO 实际会消费的检测框结果。
    result = lib_cpp.cal_result(
        feature_out[0, :, :, :],
        cfg.BOX_THRESHOLD,
        OVERLAP,
        X_MIN,
        HEIGHT,
        WIDTH,
        cfg.VOXEL_SIZE[0],
        cfg.VOXEL_SIZE[1],
        cfg.VOXEL_SIZE[2],
        cfg.NMS_THRESHOLD)
    if result is None:
        return np.empty((0, 9), dtype=np.float32)
    return np.asarray(result)


def compare_arrays(tf_result, trt_result):
    if tf_result.shape != trt_result.shape:
        return {
            "same_shape": False,
            "tf_shape": tf_result.shape,
            "trt_shape": trt_result.shape,
            "max_abs": None,
            "mean_abs": None,
        }
    if tf_result.size == 0:
        return {
            "same_shape": True,
            "tf_shape": tf_result.shape,
            "trt_shape": trt_result.shape,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }
    diff = np.abs(tf_result.astype(np.float32) - trt_result.astype(np.float32))
    return {
        "same_shape": True,
        "tf_shape": tf_result.shape,
        "trt_shape": trt_result.shape,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="对比 TensorFlow 和 TensorRT 的 LivoxDetection 输出")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", default="/velodyne_points")
    parser.add_argument("--engine", default="model/livoxmodel.engine")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    tf_runner = TfRunner()
    trt_runner = TrtRunner(args.engine)
    frame_count = 0
    checked_count = 0
    feature_max_abs = []
    feature_mean_abs = []
    detection_shape_mismatch = 0
    detection_max_abs = []

    try:
        with rosbag.Bag(args.bag, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=[args.topic]):
                if frame_count % args.stride != 0:
                    frame_count += 1
                    continue
                voxel = data2voxel(msg)
                t0 = time.time()
                tf_feature = tf_runner.infer(voxel)
                t1 = time.time()
                trt_feature = trt_runner.infer(voxel)
                t2 = time.time()

                feature_cmp = compare_arrays(tf_feature, trt_feature)
                feature_max_abs.append(feature_cmp["max_abs"])
                feature_mean_abs.append(feature_cmp["mean_abs"])

                tf_det = postprocess(tf_feature)
                trt_det = postprocess(trt_feature)
                det_cmp = compare_arrays(tf_det, trt_det)
                if not det_cmp["same_shape"]:
                    detection_shape_mismatch += 1
                else:
                    detection_max_abs.append(det_cmp["max_abs"])

                print(
                    "frame={} stamp={:.6f} tf_ms={:.2f} trt_ms={:.2f} "
                    "feature_max_abs={:.6f} feature_mean_abs={:.6f} "
                    "tf_det={} trt_det={} det_max_abs={}".format(
                        frame_count,
                        msg.header.stamp.to_sec(),
                        (t1 - t0) * 1000.0,
                        (t2 - t1) * 1000.0,
                        feature_cmp["max_abs"],
                        feature_cmp["mean_abs"],
                        tf_det.shape[0],
                        trt_det.shape[0],
                        "shape_mismatch" if not det_cmp["same_shape"] else "{:.6f}".format(det_cmp["max_abs"])))

                checked_count += 1
                frame_count += 1
                if checked_count >= args.limit:
                    break
    finally:
        tf_runner.close()
        trt_runner.close()

    print("summary_checked_frames={}".format(checked_count))
    if checked_count > 0:
        print("summary_feature_max_abs_max={:.6f}".format(max(feature_max_abs)))
        print("summary_feature_mean_abs_avg={:.6f}".format(sum(feature_mean_abs) / len(feature_mean_abs)))
        print("summary_detection_shape_mismatch={}".format(detection_shape_mismatch))
        if detection_max_abs:
            print("summary_detection_max_abs_max={:.6f}".format(max(detection_max_abs)))


if __name__ == "__main__":
    main()

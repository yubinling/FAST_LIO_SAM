#!/usr/bin/env python
import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import rosbag
import rospy

import config.config as cfg
from realtime_detection_limot import CHANNELS, HEIGHT, WIDTH, RealtimeDetector, TrtRunner


class OfflineBagDetector(object):
    def __init__(self, args):
        self.args = args
        self.helper = object.__new__(RealtimeDetector)
        self.trt_runner = None
        self.sess = None
        self.ops = None

        if args.backend == "trt":
            # 离线生成 txt 时也可以走 TRT；网络输出之后仍复用实时节点的 C++ 后处理和 txt 格式。
            self.trt_runner = TrtRunner(args.engine_path)
            self.trt_runner.infer(np.zeros(self.trt_runner.input_shape, dtype=np.float32))
            self.helper.trt_runner = self.trt_runner
            self.helper.sess = None
            self.helper.ops = None
        else:
            import tensorflow as tf
            from networks.model import livox_model

            # TensorFlow 后端保留原 checkpoint 路径，用来生成和旧离线脚本一致的检测 txt。
            net = livox_model(HEIGHT, WIDTH, CHANNELS)
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(cfg.GPU_INDEX)):
                    input_bev_img_pl = net.placeholder_inputs(cfg.BATCH_SIZE)
                    end_points = net.get_model(input_bev_img_pl)
                    saver = tf.train.Saver()
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    config.allow_soft_placement = True
                    config.log_device_placement = False
                    self.sess = tf.Session(config=config)
                    saver.restore(self.sess, cfg.MODEL_PATH)
                    self.ops = {
                        'input_bev_img_pl': input_bev_img_pl,
                        'end_points': end_points,
                    }
            self.helper.trt_runner = None
            self.helper.sess = self.sess
            self.helper.ops = self.ops

    def close(self):
        if self.trt_runner is not None:
            self.trt_runner.close()
        if self.sess is not None:
            self.sess.close()

    def detect_msg(self, cloud_msg):
        points = self.helper.read_points(cloud_msg)
        if len(points) == 0:
            boxes = []
        else:
            voxel = self.helper.data2voxel(points)
            voxel = np.expand_dims(voxel, axis=0)
            boxes = self.helper.detect(voxel)
        return self.helper.boxes_to_msg(cloud_msg.header.stamp.to_sec(), boxes)


def write_header(output_file):
    output_file.write(
        "# Frame format: [timestamp, type1, x1, y1, z1, l1, w1, h1, yaw1, score1, type2, ...]\n"
    )
    output_file.write("# Type: 0=car, 1=bus, 2=truck, 3=pedestrian, 4=bimo\n")
    output_file.write("# Coordinates: jsk frame (same as realtime_detection_limot.py output)\n")


def write_detection_line(output_file, msg):
    output_file.write("[" + ", ".join(["%.6f" % value for value in msg.data]) + "]\n")


def parse_args():
    parser = argparse.ArgumentParser(description="从 rosbag 离线生成 LIO 可读取的检测 txt")
    parser.add_argument("--bag", required=True, help="输入 rosbag 路径")
    parser.add_argument("--topic", default="/velodyne_points", help="点云话题")
    parser.add_argument("--output", required=True, help="输出 txt 路径")
    parser.add_argument("--backend", choices=("tf", "trt"), default="tf", help="检测后端")
    parser.add_argument("--engine_path", default=os.path.join(SCRIPT_DIR, "model", "livoxmodel_fp32.engine"),
                        help="TRT 后端使用的 engine 路径")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 帧；0 表示处理全部")
    parser.add_argument("--verbose", action="store_true", help="打印每帧耗时")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.bag):
        raise RuntimeError("bag 文件不存在: {}".format(args.bag))

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    rospy.init_node("livox_detection_offline_txt", anonymous=True, disable_signals=True)
    detector = OfflineBagDetector(args)
    processed = 0
    start_time = time.time()

    try:
        with rosbag.Bag(args.bag, "r") as bag, open(args.output, "w") as output_file:
            write_header(output_file)
            for _, msg, _ in bag.read_messages(topics=[args.topic]):
                frame_start = time.time()
                detect_msg = detector.detect_msg(msg)
                write_detection_line(output_file, detect_msg)
                processed += 1

                if args.verbose:
                    elapsed_ms = (time.time() - frame_start) * 1000.0
                    print("frame={} stamp={:.6f} objects={} time_ms={:.2f}".format(
                        processed - 1,
                        detect_msg.data[0],
                        int((len(detect_msg.data) - 1) / 9),
                        elapsed_ms))

                if args.limit > 0 and processed >= args.limit:
                    break
    finally:
        detector.close()

    total_time = time.time() - start_time
    print("Saved {} frames to {}".format(processed, os.path.abspath(args.output)))
    if processed > 0:
        print("Average time: {:.2f} ms/frame".format(total_time * 1000.0 / processed))


if __name__ == "__main__":
    main()

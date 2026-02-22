#!/usr/bin/env python
import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型路径在 config/config.py 中写成相对路径 "model/livoxmodel"。
# 因此无论 roslaunch 从哪里启动，都先切到检测目录，保证模型和 lib_cpp.so 可被找到。
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pcl2
import tensorflow as tf
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray

import config.config as cfg
import lib_cpp
from networks.model import livox_model


DX = cfg.VOXEL_SIZE[0]
DY = cfg.VOXEL_SIZE[1]
DZ = cfg.VOXEL_SIZE[2]

X_MIN = cfg.RANGE['X_MIN']
X_MAX = cfg.RANGE['X_MAX']
Y_MIN = cfg.RANGE['Y_MIN']
Y_MAX = cfg.RANGE['Y_MAX']
Z_MIN = cfg.RANGE['Z_MIN']
Z_MAX = cfg.RANGE['Z_MAX']

OVERLAP = cfg.OVERLAP
HEIGHT = round((X_MAX - X_MIN + 2 * OVERLAP) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)

T1 = np.array([[0.0, -1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0, 0.0],
               [1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

# 和离线 ros_limot.py 保持一致: type 数值会直接进入 LIO 的 setFrame()。
TYPE_INDICES = {'car': 0, 'bus': 1, 'truck': 2, 'pedestrian': 3, 'bimo': 4}


class RealtimeDetector(object):
    def __init__(self, args):
        self.args = args
        self.frame_count = 0
        self.debug_file = None

        if args.debug_output:
            # debug txt 只用于和离线 detectmsg/*.txt 对齐检查；LIO 实时输入仍然是 /detect3d。
            debug_dir = os.path.dirname(os.path.abspath(args.debug_output))
            if debug_dir and not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)
            self.debug_file = open(args.debug_output, 'w')
            self.debug_file.write(
                "# Frame format: [timestamp, type1, x1, y1, z1, l1, w1, h1, yaw1, score1, type2, ...]\n"
            )
            self.debug_file.write("# Type: 0=car, 1=bus, 2=truck, 3=pedestrian, 4=bimo\n")
            self.debug_file.write("# Coordinates: jsk frame (same as ros_limot.py output)\n")
            self.debug_file.flush()

        rospy.loginfo("Loading LivoxDetection TensorFlow model from %s", cfg.MODEL_PATH)
        self.net = livox_model(HEIGHT, WIDTH, CHANNELS)
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(cfg.GPU_INDEX)):
                input_bev_img_pl = self.net.placeholder_inputs(cfg.BATCH_SIZE)
                end_points = self.net.get_model(input_bev_img_pl)

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
        rospy.loginfo("LivoxDetection model loaded")

        self.pub = rospy.Publisher(args.detect_topic, Float64MultiArray, queue_size=args.queue_size)
        self.sub = rospy.Subscriber(args.cloud_topic, PointCloud2, self.cloud_callback,
                                    queue_size=args.queue_size)
        rospy.loginfo("Realtime detector subscribed to %s and publishing %s",
                      args.cloud_topic, args.detect_topic)

    def close(self):
        if self.debug_file is not None:
            self.debug_file.close()

    def roty(self, t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def get_3d_box(self, box_size, heading_angle, center):
        rot = self.roty(heading_angle)
        l, w, h = box_size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.dot(rot, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + center[0]
        corners_3d[1, :] = corners_3d[1, :] + center[1]
        corners_3d[2, :] = corners_3d[2, :] + center[2]
        return np.transpose(corners_3d)

    def data2voxel(self, pclist):
        data = [0 for _ in range(HEIGHT * WIDTH * CHANNELS)]

        for line in pclist:
            x = float(line[0])
            y = float(line[1])
            z = float(line[2])
            if y > Y_MIN and y < Y_MAX and x > X_MIN and x < X_MAX and z > Z_MIN and z < Z_MAX:
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
        return np.reshape(data, (HEIGHT, WIDTH, CHANNELS))

    def detect(self, batch_bev_img):
        feed_dict = {self.ops['input_bev_img_pl']: batch_bev_img}
        feature_out, = self.sess.run([self.ops['end_points']['feature_out']],
                                     feed_dict=feed_dict)
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

        if result is None or result.size == 0:
            return []

        is_obj_list = result[:, 0].tolist()
        reg_m_x_list = result[:, 5].tolist()
        reg_w_list = result[:, 4].tolist()
        reg_l_list = result[:, 3].tolist()
        obj_cls_list = result[:, 1].tolist()
        reg_m_y_list = result[:, 6].tolist()
        reg_theta_list = result[:, 2].tolist()
        reg_m_z_list = result[:, 8].tolist()
        reg_h_list = result[:, 7].tolist()

        boxes = []
        for i in range(len(is_obj_list)):
            box3d_pts_3d = np.ones((8, 4), float)
            box3d_pts_3d[:, 0:3] = self.get_3d_box(
                (reg_l_list[i], reg_w_list[i], reg_h_list[i]),
                reg_theta_list[i],
                (reg_m_x_list[i], reg_m_z_list[i], reg_m_y_list[i]))
            box3d_pts_3d = np.dot(np.linalg.inv(T1), box3d_pts_3d.T).T

            if int(obj_cls_list[i]) == 0:
                cls_name = 'car'
            elif int(obj_cls_list[i]) == 1:
                cls_name = 'bus'
            elif int(obj_cls_list[i]) == 2:
                cls_name = 'truck'
            elif int(obj_cls_list[i]) == 3:
                cls_name = 'pedestrian'
            else:
                cls_name = 'bimo'

            center = (reg_m_y_list[i], -reg_m_x_list[i], -reg_m_z_list[i])
            dimensions = (reg_l_list[i], reg_w_list[i], reg_h_list[i])
            # 这里的中心、尺寸、yaw 转换必须和离线 ros_limot.py 完全一致，
            # 否则实时 /detect3d 和提前生成的 txt 会产生系统性差异。
            yaw_jsk = -reg_theta_list[i] + np.pi / 2
            boxes.append({
                'label': TYPE_INDICES[cls_name],
                'score': is_obj_list[i],
                'x': center[0],
                'y': center[1],
                'z': center[2],
                'l': dimensions[0],
                'w': dimensions[1],
                'h': dimensions[2],
                'yaw': yaw_jsk,
            })
        return boxes

    def read_points(self, cloud_msg):
        points = []
        try:
            iterator = pcl2.read_points(cloud_msg, skip_nans=True,
                                        field_names=("x", "y", "z", "intensity"))
            for point in iterator:
                if point[0] == 0 and point[1] == 0 and point[2] == 0:
                    continue
                if np.abs(point[0]) < 2.0 and np.abs(point[1]) < 1.5:
                    continue
                points.append(point)
        except Exception:
            for point in pcl2.read_points(cloud_msg, skip_nans=True):
                if len(point) < 3:
                    continue
                if point[0] == 0 and point[1] == 0 and point[2] == 0:
                    continue
                if np.abs(point[0]) < 2.0 and np.abs(point[1]) < 1.5:
                    continue
                intensity = point[3] if len(point) >= 4 else 0.0
                points.append((point[0], point[1], point[2], intensity))
        return points

    def boxes_to_msg(self, timestamp, boxes):
        msg = Float64MultiArray()
        # laserMapping::setFrame() 期望格式:
        # [timestamp, type, x, y, z, l, w, h, yaw, score, ...]
        # timestamp 使用输入点云 header stamp，便于和 lidar_start_time 在 1ms 内匹配。
        msg.data.append(float(timestamp))
        for box in boxes:
            msg.data.extend([
                float(box['label']),
                float(box['x']),
                float(box['y']),
                float(box['z']),
                float(box['l']),
                float(box['w']),
                float(box['h']),
                float(box['yaw']),
                float(box['score']),
            ])
        return msg

    def write_debug_line(self, msg):
        if self.debug_file is None:
            return
        line = "[" + ", ".join(["%.6f" % value for value in msg.data]) + "]\n"
        self.debug_file.write(line)
        self.debug_file.flush()

    def cloud_callback(self, cloud_msg):
        timestamp = cloud_msg.header.stamp.to_sec()
        timer_start = time.time()
        points = self.read_points(cloud_msg)

        if len(points) == 0:
            boxes = []
        else:
            points_np = np.asarray(points)
            voxel = self.data2voxel(points_np)
            voxel = np.expand_dims(voxel, axis=0)
            boxes = self.detect(voxel)

        msg = self.boxes_to_msg(timestamp, boxes)
        # 这是 LIO 真正消费的实时检测结果；debug txt 只是旁路记录。
        self.pub.publish(msg)
        self.write_debug_line(msg)

        if self.args.verbose:
            elapsed_ms = (time.time() - timer_start) * 1000.0
            rospy.loginfo("Detection frame %d stamp %.6f: %d boxes, %.2f ms",
                          self.frame_count, timestamp, len(boxes), elapsed_ms)
        self.frame_count += 1


def parse_args():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in ("true", "1", "yes", "y", "on"):
            return True
        if value in ("false", "0", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError("boolean value expected")

    parser = argparse.ArgumentParser(description="Realtime LivoxDetection publisher for FAST-LIO-SAM")
    parser.add_argument("--cloud_topic", default="/velodyne_points")
    parser.add_argument("--detect_topic", default="/detect3d")
    parser.add_argument("--debug_output", default="")
    parser.add_argument("--queue_size", type=int, default=10)
    parser.add_argument("--verbose", nargs="?", const=True, default=False, type=str2bool)
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main():
    args = parse_args()
    rospy.init_node("realtime_detection_limot", anonymous=False)
    detector = RealtimeDetector(args)
    rospy.on_shutdown(detector.close)
    rospy.spin()


if __name__ == "__main__":
    main()

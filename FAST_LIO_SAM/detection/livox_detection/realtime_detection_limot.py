#!/usr/bin/env python
import argparse
import ctypes
import ctypes.util
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
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray

import config.config as cfg
import lib_cpp


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


class TrtRunner(object):
    def __init__(self, engine_path):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT backend requires tensorrt in the se-ssd environment."
            ) from exc

        if not os.path.isfile(engine_path):
            raise RuntimeError("TensorRT engine not found: {}".format(engine_path))

        self.trt = trt
        self.cudart = self.load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as engine_file:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine: {}".format(engine_path))

        self.context = self.engine.create_execution_context()
        self.bindings = [None] * self.engine.num_bindings
        self.input_index = None
        self.output_index = None
        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        self.device_allocations = []

        # engine 是固定 batch/固定 shape，这里提前为输入输出 binding 分配 host/GPU 内存。
        for index in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            size = trt.volume(shape)
            host_mem = np.empty(size, dtype=dtype)
            device_mem = self.cuda_malloc(host_mem.nbytes)
            self.device_allocations.append(device_mem)
            self.bindings[index] = int(device_mem.value)
            if self.engine.binding_is_input(index):
                self.input_index = index
                self.input_shape = shape
                self.host_inputs[index] = host_mem
                self.device_inputs[index] = device_mem
            else:
                self.output_index = index
                self.output_shape = shape
                self.host_outputs[index] = host_mem
                self.device_outputs[index] = device_mem

        if self.input_index is None or self.output_index is None:
            raise RuntimeError("TensorRT engine must have one input and one output")

    def load_cudart(self):
        # 直接通过 CUDA runtime 做显存申请和拷贝，避免额外引入 pycuda 依赖。
        lib_name = ctypes.util.find_library("cudart") or "libcudart.so"
        cudart = ctypes.CDLL(lib_name)
        cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cudart.cudaMalloc.restype = ctypes.c_int
        cudart.cudaFree.argtypes = [ctypes.c_void_p]
        cudart.cudaFree.restype = ctypes.c_int
        cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        cudart.cudaMemcpy.restype = ctypes.c_int
        return cudart

    def check_cuda(self, status, action):
        if status != 0:
            raise RuntimeError("CUDA runtime call failed during {}: status {}".format(action, status))

    def cuda_malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check_cuda(self.cudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def cuda_memcpy_htod(self, device_ptr, host_array):
        host_array = np.ascontiguousarray(host_array)
        self.check_cuda(
            self.cudart.cudaMemcpy(
                device_ptr,
                host_array.ctypes.data_as(ctypes.c_void_p),
                host_array.nbytes,
                1),
            "cudaMemcpy host-to-device")

    def cuda_memcpy_dtoh(self, host_array, device_ptr):
        self.check_cuda(
            self.cudart.cudaMemcpy(
                host_array.ctypes.data_as(ctypes.c_void_p),
                device_ptr,
                host_array.nbytes,
                2),
            "cudaMemcpy device-to-host")

    def close(self):
        for ptr in self.device_allocations:
            if ptr:
                self.cudart.cudaFree(ptr)
        self.device_allocations = []

    def infer(self, batch_bev_img):
        if tuple(batch_bev_img.shape) != tuple(self.input_shape):
            raise RuntimeError(
                "Unexpected TensorRT input shape: got {}, expected {}".format(
                    batch_bev_img.shape, self.input_shape))

        host_input = self.host_inputs[self.input_index]
        # 输入从 numpy 拷到 GPU，执行 TRT engine 后再把 feature_out 拷回 CPU 交给 C++ 后处理。
        np.copyto(host_input, np.ascontiguousarray(batch_bev_img).ravel())
        self.cuda_memcpy_htod(self.device_inputs[self.input_index], host_input)
        self.context.execute_v2(bindings=self.bindings)
        host_output = self.host_outputs[self.output_index]
        self.cuda_memcpy_dtoh(host_output, self.device_outputs[self.output_index])
        return np.array(host_output).reshape(self.output_shape)


class RealtimeDetector(object):
    def __init__(self, args):
        self.args = args
        self.frame_count = 0
        self.debug_file = None

        if args.debug_output:
            # 调试 txt 只用于和离线 detectmsg/*.txt 对齐检查；LIO 实时输入仍然是 /detect3d。
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

        self.trt_runner = None
        self.sess = None
        self.ops = None
        if args.backend == "trt":
            # TensorRT 只替换网络前向推理；后处理和 /detect3d 消息格式继续复用 TensorFlow 路径。
            rospy.loginfo("Loading LivoxDetection TensorRT engine from %s", args.engine_path)
            self.trt_runner = TrtRunner(args.engine_path)
            rospy.loginfo("LivoxDetection TensorRT engine loaded")
        else:
            import tensorflow as tf
            from networks.model import livox_model

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
            rospy.loginfo("LivoxDetection TensorFlow model loaded")

        self.pub = rospy.Publisher(args.detect_topic, Float64MultiArray, queue_size=args.queue_size)
        self.sub = rospy.Subscriber(args.cloud_topic, PointCloud2, self.cloud_callback,
                                    queue_size=args.queue_size)
        rospy.loginfo("Realtime detector subscribed to %s and publishing %s",
                      args.cloud_topic, args.detect_topic)

    def close(self):
        if self.debug_file is not None:
            self.debug_file.close()
        if self.trt_runner is not None:
            self.trt_runner.close()

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
        return np.reshape(data, (HEIGHT, WIDTH, CHANNELS)).astype(np.float32)

    def detect(self, batch_bev_img):
        if self.trt_runner is not None:
            feature_out = self.trt_runner.infer(batch_bev_img)
        else:
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
        # 这是 LIO 真正消费的实时检测结果；调试 txt 只是旁路记录。
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
    parser.add_argument("--backend", choices=("tf", "trt"), default="tf")
    parser.add_argument("--engine_path", default=os.path.join(SCRIPT_DIR, "model", "livoxmodel.engine"))
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main():
    args = parse_args()
    rospy.init_node("realtime_detection_limot", anonymous=False)
    detector = RealtimeDetector(args)
    rospy.on_shutdown(detector.close)
    rospy.spin()


if __name__ == "__main__":
    main()

# Realtime Detection for FAST-LIO-SAM

This folder contains the minimal LivoxDetection runtime used by
`mapping_robosense_realtime_detection.launch`.

## Runtime

The realtime detector is launched through:

```bash
roslaunch fast_lio_sam mapping_robosense_realtime_detection.launch
```

The launch starts:

- `fastlio_sam_mapping`
- `realtime_detection_limot`
- `rviz` by default

The detector wrapper activates the `se-ssd` conda environment before running
`livox_detection/realtime_detection_limot.py`.

## Data Flow

```text
/velodyne_points -> realtime_detection_limot -> /detect3d -> laserMapping
```

`/detect3d` uses the same `std_msgs/Float64MultiArray` layout as the existing
txt files:

```text
[timestamp, type, x, y, z, l, w, h, yaw, score, ...]
```

The timestamp is copied from the input point cloud header so it can match
`laserMapping` within the existing 1 ms tolerance.

## Default Check Bag

Use this bag for the first txt-vs-realtime comparison:

```bash
rosbag play /home/lingyubin/Documents/lyb/bag/satslam_realcar_Data/fast4_withcar.bag
```

By default, realtime detections are also written to:

```text
FAST_LIO_SAM/detection/debug/fast4_realtime.txt
```

Compare that file with the existing `fast4.txt` detection file to check output
compatibility.

## TensorRT Preparation

The first TensorRT preparation step is exporting the TF1 checkpoint to a frozen
graph:

```bash
FAST_LIO_SAM/scripts/run_realtime_detection_limot.sh \
  --help

cd FAST_LIO_SAM/detection/livox_detection
conda run -n se-ssd python export_frozen_graph.py
```

The exported graph uses stable tensor names:

```text
input:  input_bev_img:0  dtype=float32  shape=[1, 1008, 224, 30]
output: feature_out:0
```

The current `se-ssd` environment has TensorFlow 1.15, but does not yet include
`onnx`, `tf2onnx`, or `tensorrt`. Those are needed for the next conversion step
from frozen graph/ONNX to TensorRT engine.

## TensorRT 实时运行

生成并验证 `model/livoxmodel_fp32.engine` 之后，可以用 TensorRT 后端启动实时检测：

```bash
roslaunch fast_lio_sam mapping_robosense_realtime_detection.launch \
  detection_backend:=trt
```

这里只替换网络前向推理，`lib_cpp.cal_result` 后处理和 `/detect3d` 消息格式仍然和 TensorFlow
后端保持一致。

TensorRT Python 后端需要 `se-ssd` 环境里有 `tensorrt`，并且能找到
`/usr/local/TensorRT-8.5.3.1` 下的 TensorRT/CUDA 运行库。GPU 内存拷贝直接调用
`libcudart`，不依赖 `pycuda`。

在同一段 bag 上对比 TensorFlow 和 TensorRT：

```bash
cd FAST_LIO_SAM/detection/livox_detection
TENSORRT_DIR=/usr/local/TensorRT-8.5.3.1 \
LD_LIBRARY_PATH=$TENSORRT_DIR/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH \
conda run -n se-ssd python compare_detection_backends.py \
  --bag /home/lingyubin/Documents/lyb/bag/satslam_realcar_Data/fast4_withcar.bag \
  --topic /velodyne_points \
  --engine model/livoxmodel_fp32.engine \
  --limit 10
```

当前 FP32 engine 在 `fast4_withcar.bag` 前 10 帧上验证通过：检测框数量一致，
`summary_detection_shape_mismatch=0`，最大检测框数值差异约为 `0.000858`。之前测试过的
FP16 engine 会出现 `inf` 输出，不建议作为默认运行路径。

## 离线生成 txt

不启动 LIO、只从 bag 生成检测 txt，可以直接使用当前包里的离线脚本：

```bash
FAST_LIO_SAM/scripts/run_offline_detection_limot.sh \
  --bag /home/lingyubin/Documents/lyb/bag/satslam_realcar_Data/fast4_withcar.bag \
  --topic /velodyne_points \
  --output /home/lingyubin/Documents/lyb/livo/sat-slam/src/FAST_LIO_SAM/FAST_LIO_SAM/detectmsg/fast4.txt \
  --backend trt \
  --verbose
```

生成的 txt 格式和 LIO 的 `detect_file_path` 读取格式一致：

```text
[timestamp, type, x, y, z, l, w, h, yaw, score, ...]
```

`--backend tf` 使用 TensorFlow checkpoint，`--backend trt` 使用
`model/livoxmodel_fp32.engine`。TRT 模式不会主动加载 TensorFlow，适合快速预生成 txt。
可以加 `--limit 10` 只处理前 10 帧做测试。

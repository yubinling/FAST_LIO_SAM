# LivoxDetection 检测模块说明

本目录保存 SAT-SLAM 使用的本地 LivoxDetection 推理代码，支持以下四种用途：

- TensorFlow 实时检测并发布 `/detect3d`。
- TensorRT 实时检测并发布 `/detect3d`。
- TensorFlow 离线读取 rosbag 并生成检测 txt。
- TensorRT 离线读取 rosbag 并生成检测 txt。

## 1. 目录结构

```text
detection/
├── README.md
├── debug/                         # 实时检测旁路记录的 txt
└── livox_detection/
    ├── config/config.py           # 网络范围、体素、阈值和 checkpoint 配置
    ├── networks/model.py          # TensorFlow 网络定义
    ├── model/
    │   ├── livoxmodel.data-00000-of-00001
    │   ├── livoxmodel.index
    │   ├── livoxmodel.meta
    │   ├── livoxmodel_frozen.pb
    │   ├── livoxmodel.onnx
    │   └── livoxmodel_fp32.engine
    ├── lib_cpp.so                 # 检测结果 C++ 后处理
    ├── realtime_detection_limot.py
    ├── offline_detection_to_txt.py
    ├── export_frozen_graph.py
    └── compare_detection_backends.py
```

外层启动脚本：

```text
FAST_LIO_SAM/scripts/run_realtime_detection_limot.sh
FAST_LIO_SAM/scripts/run_offline_detection_limot.sh
```

这两个 shell 脚本都会主动执行：

```bash
conda activate se-ssd
```

因此即使从 base 环境运行 roslaunch，检测节点仍会进入 `se-ssd` 环境。

## 2. 检测输出格式

实时话题和离线 txt 使用相同数据格式：

```text
[timestamp, type, x, y, z, l, w, h, yaw, score, ...]
```

类别编号：

```text
0: car
1: bus
2: truck
3: pedestrian
4: bimo
```

实时模式使用输入点云 `header.stamp` 作为检测时间戳，便于
`laserMapping` 将 `/detect3d` 和当前雷达帧对齐。

## 3. 实时检测

### 3.1 TensorFlow

```bash
roslaunch fast_lio_sam mapping_robosense_realtime_detection.launch \
  detection_backend:=tf
```

TensorFlow 路径读取 checkpoint：

```text
model/livoxmodel.data-00000-of-00001
model/livoxmodel.index
model/livoxmodel.meta
```

### 3.2 TensorRT

```bash
roslaunch fast_lio_sam mapping_robosense_realtime_detection.launch \
  detection_backend:=trt
```

默认 engine：

```text
model/livoxmodel_fp32.engine
```

指定其他 engine：

```bash
roslaunch fast_lio_sam mapping_robosense_realtime_detection.launch \
  detection_backend:=trt \
  detection_engine:=/absolute/path/to/model.engine
```

检测较慢、LIO 等不到同时间戳结果时，可临时增加等待时间：

```bash
roslaunch fast_lio_sam mapping_robosense_realtime_detection.launch \
  detection_backend:=trt \
  detection_wait_ms:=500
```

TensorRT 只替换网络前向推理。体素构建、`lib_cpp.cal_result` 后处理、
类别编号和 `/detect3d` 格式均与 TensorFlow 路径一致。

## 4. 离线生成 txt

### 4.1 `run_offline` 是否同时支持 TF 和 TRT

是。`run_offline_detection_limot.sh` 会把参数原样传给
`offline_detection_to_txt.py`，其中：

```text
--backend tf   使用 TensorFlow checkpoint
--backend trt  使用 TensorRT engine
```

两种后端生成相同格式的 txt，都可以被 `detect_file_path` 读取。

### 4.2 TensorFlow 离线生成

```bash
FAST_LIO_SAM/scripts/run_offline_detection_limot.sh \
  --bag /path/to/data.bag \
  --topic /velodyne_points \
  --output /path/to/detectmsg/data_tf.txt \
  --backend tf \
  --verbose
```

### 4.3 TensorRT 离线生成

```bash
FAST_LIO_SAM/scripts/run_offline_detection_limot.sh \
  --bag /path/to/data.bag \
  --topic /velodyne_points \
  --output /path/to/detectmsg/data_trt.txt \
  --backend trt \
  --engine_path FAST_LIO_SAM/detection/livox_detection/model/livoxmodel_fp32.engine \
  --verbose
```

只处理前 10 帧：

```bash
FAST_LIO_SAM/scripts/run_offline_detection_limot.sh \
  --bag /path/to/data.bag \
  --output /tmp/test_detection.txt \
  --backend trt \
  --limit 10
```

说明：

- TRT 模式不会主动加载 TensorFlow。
- TF 模式不需要 engine。
- `--limit 0` 表示处理全部帧。
- `--verbose` 会打印每帧耗时、时间戳和目标数量。

## 5. 从 checkpoint 生成 TensorRT engine

当前推荐路线：

```text
TensorFlow checkpoint
  -> frozen.pb
  -> ONNX
  -> TensorRT FP32 engine
  -> TF/TRT 结果对比
```

### 5.1 环境检查

```bash
conda activate se-ssd

python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import onnx; print(onnx.__version__)"
python -c "import tf2onnx; print(tf2onnx.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"
```

当前工程使用过的 Python 3.6 兼容版本：

```text
onnx==1.10.2
tf2onnx==1.9.3
protobuf<4
```

缺少 ONNX 工具时，可在 `se-ssd` 环境中安装：

```bash
python -m pip install \
  --index-url https://pypi.org/simple \
  "protobuf<4" \
  "onnx==1.10.2" \
  "tf2onnx==1.9.3"
```

如果 TensorRT 来自本地压缩包安装，先设置：

```bash
export TENSORRT_DIR=/usr/local/TensorRT-8.5.3.1
export PATH=$TENSORRT_DIR/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_DIR/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
```

若 `python -c "import tensorrt"` 失败，需要在 `se-ssd` 环境中安装
`$TENSORRT_DIR/python/` 下与 Python 3.6 对应的 TensorRT wheel。

### 5.2 checkpoint 导出 frozen PB

```bash
cd /home/lingyubin/Documents/lyb/livo/sat-slam/src/FAST_LIO_SAM/FAST_LIO_SAM/detection/livox_detection
conda activate se-ssd

python export_frozen_graph.py \
  --checkpoint model/livoxmodel \
  --output model/livoxmodel_frozen.pb
```

固定张量名：

```text
输入: input_bev_img:0
类型: float32
形状: [1, 1008, 224, 30]

输出: feature_out:0
```

### 5.3 frozen PB 转 ONNX

```bash
python -m tf2onnx.convert \
  --graphdef model/livoxmodel_frozen.pb \
  --inputs input_bev_img:0 \
  --outputs feature_out:0 \
  --opset 11 \
  --output model/livoxmodel.onnx
```

检查 ONNX：

```bash
python -c "import onnx; m=onnx.load('model/livoxmodel.onnx'); onnx.checker.check_model(m); print('ONNX OK')"
```

### 5.4 ONNX 转 FP32 engine

```bash
export TENSORRT_DIR=/usr/local/TensorRT-8.5.3.1
export PATH=$TENSORRT_DIR/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_DIR/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH

$TENSORRT_DIR/bin/trtexec \
  --onnx=model/livoxmodel.onnx \
  --saveEngine=model/livoxmodel_fp32.engine \
  --workspace=2048
```

当前默认使用 FP32，不添加 `--fp16`。之前测试的 FP16 engine 出现过 `inf`
输出，因此不建议直接替换默认 engine。

注意：

- TensorRT engine 和 GPU 架构、TensorRT/CUDA 版本相关。
- 更换显卡、TensorRT 或 CUDA 后，建议从 ONNX 重新生成 engine。
- 不要只根据 engine 能加载就认为结果正确，必须做 TF/TRT 检测框对比。

## 6. 验证 TF 和 TRT 一致性

```bash
cd /home/lingyubin/Documents/lyb/livo/sat-slam/src/FAST_LIO_SAM/FAST_LIO_SAM/detection/livox_detection

TENSORRT_DIR=/usr/local/TensorRT-8.5.3.1 \
LD_LIBRARY_PATH=$TENSORRT_DIR/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH \
conda run -n se-ssd python compare_detection_backends.py \
  --bag /path/to/data.bag \
  --topic /velodyne_points \
  --engine model/livoxmodel_fp32.engine \
  --limit 10
```

重点查看：

```text
summary_detection_shape_mismatch
summary_detection_max_abs_max
summary_feature_max_abs_max
```

理想结果：

- `summary_detection_shape_mismatch=0`
- TF/TRT 检测框数量一致
- 检测框最大数值差异足够小
- 输出中没有 `nan` 或 `inf`

当前 FP32 engine 曾在 `fast4_withcar.bag` 前 10 帧验证通过，检测框数量一致，
最大检测框数值差异约为 `0.000858`。

## 7. txt 与实时结果对比

实时检测可以旁路记录 txt：

```text
detection/debug/fast4_realtime.txt
```

该文件只用于检查，不是 LIO 的实时输入。实时 LIO 实际消费的是 `/detect3d`。

离线和实时结果应重点对比：

- 时间戳是否来自同一帧点云。
- 每帧目标数量是否一致。
- 类型、中心、尺寸、yaw 和 score 是否一致。
- 是否有检测结果晚于 LIO 等待窗口。

## 8. 常见问题

### 8.1 在 base 环境启动为什么仍能检测

ROS 启动的是：

```text
scripts/run_realtime_detection_limot.sh
```

脚本内部执行 `conda activate se-ssd`，因此检测进程使用的是 `se-ssd`，
不取决于启动 roslaunch 的终端当前是不是 base。

### 8.2 TRT 模式会不会加载 TensorFlow

不会。实时和离线脚本都只在 `backend == "tf"` 时导入 TensorFlow。

### 8.3 engine 文件在哪里

默认路径：

```text
detection/livox_detection/model/livoxmodel_fp32.engine
```

实时 launch 使用 `detection_engine` 指定，离线脚本使用 `--engine_path` 指定。

### 8.4 `Waiting for test results more than ...`

表示 LIO 等待同时间戳检测结果超过配置时间。可检查：

- 检测是否订阅正确点云话题。
- `/detect3d` 时间戳是否来自输入点云 header。
- 检测耗时是否大于 `limot/detection_wait_ms`。
- rosbag 播放速度是否过快。

### 8.5 TRT engine 加载失败

检查：

```bash
echo $TENSORRT_DIR
echo $LD_LIBRARY_PATH
python -c "import tensorrt as trt; print(trt.__version__)"
nvidia-smi
```

如果 engine 是在另一台机器或另一版本 TensorRT 上生成的，应重新转换。

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
input:  input_bev_img:0
output: feature_out:0
```

The current `se-ssd` environment has TensorFlow 1.15, but does not yet include
`onnx`, `tf2onnx`, or `tensorrt`. Those are needed for the next conversion step
from frozen graph/ONNX to TensorRT engine.

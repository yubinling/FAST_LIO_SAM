#!/usr/bin/env python
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tensorflow as tf

import config.config as cfg
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


def parse_args():
    parser = argparse.ArgumentParser(description="Export LivoxDetection TF1 checkpoint to frozen pb")
    parser.add_argument("--checkpoint", default=cfg.MODEL_PATH,
                        help="checkpoint prefix, default: model/livoxmodel")
    parser.add_argument("--output", default="model/livoxmodel_frozen.pb",
                        help="output frozen graph path")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    graph = tf.Graph()
    with graph.as_default():
        net = livox_model(HEIGHT, WIDTH, CHANNELS)
        input_bev_img = tf.placeholder(
            tf.bool,
            shape=(cfg.BATCH_SIZE, HEIGHT, WIDTH, CHANNELS),
            name="input_bev_img")
        end_points = net.get_model(input_bev_img)
        feature_out = tf.identity(end_points["feature_out"], name="feature_out")

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, args.checkpoint)
            frozen = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                ["feature_out"])

            with tf.gfile.GFile(args.output, "wb") as output_file:
                output_file.write(frozen.SerializeToString())

    print("Exported frozen graph: {}".format(os.path.abspath(args.output)))
    print("Input tensor: input_bev_img:0")
    print("Output tensor: feature_out:0")


if __name__ == "__main__":
    main()

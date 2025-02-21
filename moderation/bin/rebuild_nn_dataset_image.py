#!/usr/bin/python3 -W ignore
"""
rebuild_nn_dataset_image.py
"""

import os
import sys
import argparse

NOMEROFF_NET_DIR = os.path.abspath('../../')
sys.path.append(NOMEROFF_NET_DIR)

from nomeroff_net.tools.nn_numberplate_dataset import DatasetConfig, DatasetItem

parser = argparse.ArgumentParser(description='Image number plates data markup rebuilder')
parser.add_argument('-anb_key', dest="anb_key", required=True,
                    help='Anb key for loadin image settings from  anb/<anb_key>.json')
parser.add_argument('-dataset_dir', dest="dataset_dir", required=True, help='Path to datset files')
parser.add_argument('-rotate', dest="rotate", required=False, help='Rotate param', default=0, type=int)
parser.add_argument('-update_lines', dest="update_lines", required=False, help='Check update lines', default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('-debug', dest="debug", required=False, help='Debug mode', default=False,
                    action=argparse.BooleanOptionalAction)
args = parser.parse_args()

dataset_dir = args.dataset_dir
rotate = args.rotate
update_lines = args.update_lines
debug = args.debug
dataset_config = DatasetConfig(dataset_dir)
anb_key = args.anb_key
dataset_item = DatasetItem(dataset_config, anb_key, rotate, update_lines, debug)
# dataset_item = DatasetItem(dataset_config, anb_key)
dataset_item.recheck_regions()


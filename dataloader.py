import json, os
import sys
import torch
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('DATA_ROOT', default='../../data/data', help='Location to root directory for dataset reading') # /mnt/mars-fast/datasets/
#parser.add_argument('SAVE_ROOT', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
parser.add_argument('--DATASET', default='road', 
                    type=str,help='dataset being used')
parser.add_argument('--TRAIN_SUBSETS', default='train_3,', 
                    type=str,help='Training SUBSETS seprated by ,')
parser.add_argument('--VAL_SUBSETS', default='', 
                    type=str,help='Validation SUBSETS seprated by ,')
parser.add_argument('--TEST_SUBSETS', default='', 
                    type=str,help='Testing SUBSETS seprated by ,')
parser.add_argument('--MODE', default='train',
                    help='MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes')
parser.add_argument('--SEQ_LEN', default=8,
                    type=int, help='Number of input frames')
parser.add_argument('-b','--BATCH_SIZE', default=4, 
                    type=int, help='Batch size for training')
parser.add_argument('--MIN_SEQ_STEP', default=1,
                    type=int, help='DIFFERENCE of gap between the frames of sequence')
parser.add_argument('--MAX_SEQ_STEP', default=1,
                    type=int, help='DIFFERENCE of gap between the frames of sequence')

args = parser.parse_args()
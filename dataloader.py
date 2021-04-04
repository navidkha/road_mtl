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


class VideoDataset(tutils.data.Dataset):
    """
    ROAD Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, input_type='rgb', skip_step=1):

        self.SUBSETS = args.TRAIN_SUBSETS
        self.SEQ_LEN = args.SEQ_LEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        self.skip_step = args.SEQ_LEN
        self.num_steps = max(1, int(self.MAX_SEQ_STEP - self.MIN_SEQ_STEP + 1 )//2)
        self.input_type = input_type+'-images'
        self.root = args.DATA_ROOT + '/'
        self._imgpath = os.path.join(self.root, self.input_type)
        self.ids = list()
        self._make_lists_road()

    def _make_lists_road(self):

        self.anno_file  = os.path.join(self.root, 'road_trainval_v1.0.json')

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']

        self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        num_label_type = 5
        self.num_classes = 1 
        self.num_classes_list = [1]
        for name in self.label_types: 
            numc = len(final_annots[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = final_annots['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)
        
        counts = np.zeros((len(final_annots[self.label_types[-1] + '_labels']), num_label_type), dtype=np.int32)
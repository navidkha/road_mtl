import json, os
import sys
import torch
import argparse
import numpy as np
from torchvision import transforms
import torch.utils as tutils
from random import shuffle


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


def is_part_of_subsets(split_ids, SUBSETS):
    
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    
    return is_it

def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids


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

        self.video_list = []
        self.numf_list = []
        frame_level_list = []

        for videoname in sorted(database.keys()):
            
            # if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
            #     continue
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]

                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], box
                        assert width==1280 and height==960, (width, height, box)

                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], final_annots[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc] = 1
                                box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 

                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes 
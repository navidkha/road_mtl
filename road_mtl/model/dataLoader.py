
import json
import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch.utils as tutils
import torch.utils.data as data_utils
from PIL import Image
from torchvision import transforms

from tasks.resnet import ResNet
from tasks.taskCreator import TaskCreator

import model.transforms as vtf
from tasks.taskNames import VisionTaskName
from utils.imageUtils import  draw_text_box
from utils.printUtility import print_warn

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224




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

    def __init__(self, args, input_type='rgb', skip_step=1, train=True):

        self._labels_definition = {}

        self.SUBSETS = args.SUBSETS
        self.SEQ_LEN = args.SEQ_LEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        # self.MULIT_SCALE = args.MULIT_SCALE
        self.skip_step = args.SEQ_LEN
        self.num_steps = max(1, int(self.MAX_SEQ_STEP - self.MIN_SEQ_STEP + 1 )//2)
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.root = args.DATA_ROOT + '/'
        self.train = train
        self._imgpath = os.path.join(self.root, self.input_type)
        # self.image_sets = image_sets

        self._mean = args.MEANS
        self._std = args.STDS
        self._is_debug_mode = args.DEBUG
        self.ids = list()  
        self._make_lists_road()
        self.transform = self._get_train_transform()

    def get_labels_definition(self):
        return self._labels_definition

    def load_labels_definition(self, final_annots):

        agent_labels = final_annots["agent_labels"]
        self._labels_definition[VisionTaskName.ActiveAgentDetection.value] = agent_labels

        action_labels = final_annots["action_labels"]
        self._labels_definition[VisionTaskName.ActionDetection.value] = action_labels

        loc_labels = final_annots["loc_labels"]
        self._labels_definition[VisionTaskName.LocationDetection.value] = loc_labels

        duplex_labels = final_annots["duplex_labels"]
        self._labels_definition[VisionTaskName.InAgentActionDetection.value] = duplex_labels
        
        triplet_labels = final_annots["triplet_labels"]
        self._labels_definition[VisionTaskName.RoadEventDetection.value] = triplet_labels

    def _make_lists_road(self):

        self.anno_file  = os.path.join(self.root, 'road_trainval_v1.0.json')

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)

        self.load_labels_definition(final_annots)

        database = final_annots['db']
        
        self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        
        num_label_type = 5
        self.num_classes = 1 ## one for presence
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
            
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                 continue
            # print(videoname)

            dir_list = []

            if self._is_debug_mode == True:
                dir_list = ["2014-06-25-16-45-34_stereo_centre_02"]

            if self._is_debug_mode == False or (self._is_debug_mode == True and videoname in dir_list):
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

                frame_level_list.append(frame_level_annos) 

                ## make ids
                start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, -1,  -self.skip_step)]
                for frame_num in start_frames:
                    step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                    shuffle(step_list)
                    # print(len(step_list), self.num_steps)
                    for s in range(min(self.num_steps, len(step_list))):
                        video_id = self.video_list.index(videoname)
                        self.ids.append([video_id, frame_num ,step_list[s]])
        # pdb.set_trace()
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = final_annots[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): 
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))

        self.label_types = ['agent_ness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_info = self.ids[index]
        video_id, start_frame, step_size = id_info
        videoname = self.video_list[video_id]
        images = []
        frame_num = start_frame
        ego_labels = np.zeros(self.SEQ_LEN)-1
        all_boxes = []
        labels = []
        ego_labels = []
        mask = np.zeros(self.SEQ_LEN, dtype=np.int)
        # indexs = []
        for i in range(self.SEQ_LEN):
            img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num+1)

            img = Image.open(img_name).convert('RGB')
            images.append(img)
            if self.frame_level_list[video_id][frame_num]['labeled']:
                mask[i] = 1
                all_boxes.append(self.frame_level_list[video_id][frame_num]['boxes'].copy())
                labels.append(self.frame_level_list[video_id][frame_num]['labels'].copy())
                ego_labels.append(self.frame_level_list[video_id][frame_num]['ego_label'])
            else:
                all_boxes.append(np.asarray([]))
                labels.append(np.asarray([]))
                ego_labels.append(-1)            
            frame_num += step_size

        clip = self.transform(images)
        height, width = clip.shape[-2:]
        wh = [height, width]
        clip = clip.view(3*self.SEQ_LEN,IMAGE_HEIGHT,IMAGE_WIDTH)
        # print(clip.shape)
        return clip, all_boxes, labels, ego_labels, index, wh, self.num_classes

    def _get_train_transform(self):
        train_transform = transforms.Compose([
                            vtf.ResizeClip(IMAGE_HEIGHT, IMAGE_WIDTH),
                            vtf.ToTensorStack(),
                            vtf.Normalize(mean=self._mean, std=self._std)])
        return train_transform


import torch
import matplotlib.pyplot as ply
if __name__ == "__main__":
    adr = "/home/mohsen/pic.jpg"
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]
    img = Image.open(adr).convert('RGB')
    images = []
    images.append(img)
    train_transform = transforms.Compose([
        vtf.ResizeClip(IMAGE_HEIGHT, IMAGE_WIDTH),
        vtf.ToTensorStack(),
        vtf.Normalize(mean=_mean, std=_std)])
    clip = train_transform(images)
    clip = clip.view(3 * 1, IMAGE_HEIGHT, IMAGE_WIDTH)

    sz = 224
    seq_len = 1
    img_new = torch.zeros([3, sz, sz])
    img_new[0] = clip[seq_len - 1]
    img_new[1] = clip[2 * seq_len - 1]
    img_new[2] = clip[3 * seq_len - 1]

    b_p = [[50,100,70,150]]
    b_r = [[10,20,30,70]]
    ig = draw_text_box(img_new, ["pred1"], ["lbl1"], b_p, b_r)
    plt.imshow(ig)
    plt.show()




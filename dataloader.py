import json, os
import sys
import torch
import argparse
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
import torch.utils as tutils
from random import shuffle
from PIL import Image, ImageDraw
import torch.utils.data as data_utils
import utils
import transforms as vtf
import functools
import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
RESIZE_SIZE = 256

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('DATA_ROOT', help='Location to root directory for dataset reading') # /mnt/mars-fast/datasets/
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
parser.add_argument('--MIN_SIZE', default=512, 
                    type=int, help='Input Size for FPN')

args = parser.parse_args()
args = utils.set_args(args) # set SUBSETS of datasets

def parse_input_line(line, dataset_root):
    """
    Parse dataset line and return image and label
    """
    # Parse the line -> subpath, label
    record_default = [[''], [0]]
    parsed_entries = tf.decode_csv(line, record_default, field_delim=' ')
    image_path = dataset_root + parsed_entries[0] # String tensors can be concatenated by add operator
    label = tf.cast(parsed_entries[1], tf.int32)

    # Read image
    raw_jpeg = tf.read_file(image_path)
    image = tf.image.decode_jpeg(raw_jpeg, channels=3)

    return image, label

def resize_image(input_image, random_aspect=False):
    # Resize image so that the shorter side is 256
    height_orig = tf.shape(input_image)[0]
    width_orig = tf.shape(input_image)[1]
    ratio_flag = tf.greater(height_orig, width_orig)  # True if height > width
    if random_aspect:
        aspect_ratio = tf.random_uniform([], minval=0.875, maxval=1.2, dtype=tf.float64)
        height = tf.where(ratio_flag, tf.cast(RESIZE_SIZE*height_orig/width_orig*aspect_ratio, tf.int32), RESIZE_SIZE)
        width = tf.where(ratio_flag, RESIZE_SIZE, tf.cast(RESIZE_SIZE*width_orig/height_orig*aspect_ratio, tf.int32))
    else:
        height = tf.where(ratio_flag, tf.cast(RESIZE_SIZE*height_orig/width_orig, tf.int32), RESIZE_SIZE)
        width = tf.where(ratio_flag, RESIZE_SIZE, tf.cast(RESIZE_SIZE*width_orig/height_orig, tf.int32))
    image = tf.image.resize_images(input_image, [height, width])
    return image

def random_sized_crop(input_image):
    # Input image -> crop with random size and random aspect ratio
    height_orig = tf.cast(tf.shape(input_image)[0], tf.float64)
    width_orig = tf.cast(tf.shape(input_image)[1], tf.float64)

    aspect_ratio = tf.random_uniform([], minval=0.75, maxval=1.33, dtype=tf.float64)
    height_max = tf.minimum(height_orig, width_orig*aspect_ratio)
    height_crop = tf.random_uniform([], minval=tf.minimum(height_max, tf.maximum(0.5*height_orig, 0.5*height_max))
                                    , maxval=height_max, dtype=tf.float64)
    width_crop = height_crop / aspect_ratio
    height_crop = tf.cast(height_crop, tf.int32)
    width_crop = tf.cast(width_crop, tf.int32)

    crop = tf.random_crop(input_image, [height_crop, width_crop, 3])

    # Resize to 224x224
    image = tf.image.resize_images(crop, [IMAGE_HEIGHT, IMAGE_WIDTH])

    return image

def lighting(input_image):
    # Lighting noise (AlexNet-style PCA-based noise) from torch code
    # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua
    alphastd = 0.1
    eigval = np.array([0.2175, 0.0188, 0.0045], dtype=np.float32)
    eigvec = np.array([[-0.5675,  0.7192,  0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948,  0.4203]], dtype=np.float32)

    alpha = tf.random_normal([3, 1], mean=0.0, stddev=alphastd)
    rgb = alpha * (eigval.reshape([3, 1]) * eigvec)
    image = input_image + tf.reduce_sum(rgb, axis=0)
    return image

def preprocess(image, label, distortion, center_crop):
    # Image augmentation
    if not distortion:
        # Resize_image
        image = resize_image(image)

        # Crop(random/center)
        height = IMAGE_HEIGHT
        width = IMAGE_WIDTH
        if not center_crop:
            image = tf.random_crop(image, [height, width, 3])
        else:
            image_shape = tf.shape(image)
            h_offset = tf.cast((image_shape[0]-height)/2, tf.int32)
            w_offset = tf.cast((image_shape[1]-width)/2, tf.int32)
            image = tf.slice(image, [h_offset, w_offset, 0], [height, width, 3])
    else:
        # Image augmentation for training the network. Note the many random distortions applied to the image.
        image = random_sized_crop(image)

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Because these operations are not commutative, consider randomizing the order their operation.
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

        # Lighting noise
        image = lighting(image)

    # Preprocess: imagr normalization per channel
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    image = (image - imagenet_mean) / imagenet_std

    return image, label

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

    def __init__(self, args, input_type='rgb', skip_step=1, train=True, transform=None):

        self.SUBSETS = 'train'#args.SUBSETS
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
        self.ids = list()  
        self._make_lists_road()
        self.transform = transform

    def _make_lists_road(self):

        self.anno_file  = os.path.join(self.root, 'road_trainval_v1.0.json')

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
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
        print(self.frame_level_list)
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


        return clip, all_boxes, labels, ego_labels, index, wh, self.num_classes
    


    def inputs_base(self, shuffle=False, num_threads=60, num_sets=1, center_crop=False):
        """Construct input for IMAGENET training/evaluation using the Reader ops.
        Args:
            dataset_root: Path to the root of ROAD datasets
            txt_fpath: Path to the txt file including image subpaths and labels
            batch_size: Number of images per batch(per set).
            num_sets: Number of sets. Note that the images are prefetched to GPUs.
        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        dataset_root = self._imgpath
        txt_fpath = self.anno_file
        batch_size = self.SEQ_LEN
        # print('\tLoad file list from %s: Total %d files' % (txt_fpath, num_examples_per_epoch))
        # print('\t\tBatch size: %d, %d sets of batches, %d threads per batch' % (batch_size, num_sets, num_threads))

        # Read txt file containing image filepaths and labels
        dataset = tf.data.TextLineDataset([txt_fpath])
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
        else:
            dataset = dataset.repeat()

        # Read txt line and load image and label
        dataset_root_t = tf.constant(dataset_root)
        parse_map = functools.partial(parse_input_line, dataset_root=dataset_root_t)
        dataset = dataset.map(parse_map)
        # dataset = dataset.map(parse_map, num_parallel_calls=num_threads)

        # Preprocess images
        images_list, labels_list = [], []
        for i in range(num_sets):
            preprocess_map = functools.partial(preprocess, distortion=False, center_crop=center_crop)
            dataset_set = dataset.apply(tf.contrib.data.map_and_batch(preprocess_map, batch_size, num_threads))

            # dataset_set = dataset_set.prefetch(10)
            dataset_set = dataset_set.apply(tf.contrib.data.prefetch_to_device('/GPU:%d'%i))
            iterator = dataset_set.make_one_shot_iterator()
            images, labels = iterator.get_next()
            images.set_shape((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            labels.set_shape((batch_size, ))

            images_list.append(images)
            labels_list.append(labels)
            print(images_list)
            print(labels_list)
        return images_list, labels_list


if args.MODE in ['train','val']:
    # args.SUBSETS = args.TRAIN_SUBSETS
    # train_transform = transforms.Compose([
    #                     vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
    #                     vtf.ToTensorStack(),
    #                     vtf.Normalize(mean=args.MEANS, std=args.STDS)])
    train_dataset = VideoDataset(args)
    train_dataset.inputs_base()


    ## For validation set
    full_test = False
    args.SUBSETS = args.VAL_SUBSETS
    skip_step = args.SEQ_LEN*8

else:
    args.MAX_SEQ_STEP = 1
    args.SUBSETS = args.TEST_SUBSETS
    full_test = True #args.MODE != 'train'

# validation set
# val_dataset = VideoDataset(args, train=False, skip_step=skip_step)


#print(train_dataset.__len__())
#clip, all_boxes, labels, ego_labels, index, wh, num_classes = train_dataset.__getitem__(1)
#print(clip)




import comet_ml
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from model.dataLoader import VideoDataset
from model.datasets import custum_collate
from tasksManager import TasksManager
from utils import ioUtils as utils
import argparse

from utils.printUtility import print_info, print_warn

import matplotlib.pylab as plt
from PIL import Image, ImageFont
from PIL import ImageDraw
def draw_text(img, text):
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf", size=36)
    ImageDraw.Draw(
        img  # Image
    ).text(
        (100, 100),  # Coordinates
        text,  # Text
        (255, 0, 0),  # Color
        font=font
    )
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":

    # img = Image.open("/home/mohsen/Downloads/11.png")
    # draw_text(img, "Mohammad")



    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('DATA_ROOT', help='Location to root directory for dataset reading')  # /mnt/mars-fast/datasets/
    # parser.add_argument('SAVE_ROOT', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
    parser.add_argument('--DATASET', default='road',
                        type=str, help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_3,',
                        type=str, help='Training SUBSETS separated by ,')
    parser.add_argument('--VAL_SUBSETS', default='val_3',
                        type=str, help='Validation SUBSETS separated by ,')
    parser.add_argument('--TEST_SUBSETS', default='',
                        type=str, help='Testing SUBSETS separated by ,')
    parser.add_argument('--MODE', default='train',
                        help='MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes')
    parser.add_argument('--SEQ_LEN', default=8,
                        type=int, help='Number of input frames')
    parser.add_argument('-b', '--BATCH_SIZE', default=8,
                        type=int, help='Batch size for training')
    parser.add_argument('--MIN_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MAX_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MIN_SIZE', default=512,
                        type=int, help='Input Size for FPN')
    parser.add_argument('--NUM_WORKERS', default=4,
                        type=int, help='Number of threads')
    parser.add_argument('--DEBUG', default=False,
                        type=bool, help='Is debug mode')
    parser.add_argument('--MULTI', default=False,
                        type=bool, help='Is multi task learning mode')

    args = parser.parse_args()
    args = utils.set_args(args)  # set SUBSETS of datasets

    if args.DEBUG:
        print_warn("-------------- Running in DEBUG mode --------------")

    if args.MODE in ['train', 'val']:
        print_info("Loading dataset ...")

        args.SUBSETS = args.TRAIN_SUBSETS
        data_set_train = VideoDataset(args, train=True)
        #print_info("Dataset loaded.")

        data_loader_train = DataLoader(dataset=data_set_train,
                                 batch_size=args.BATCH_SIZE,
                                 num_workers=args.NUM_WORKERS,
                                 shuffle=True, pin_memory=True,
                                 collate_fn=custum_collate,
                                 drop_last=True)

        args.SUBSETS = args.VAL_SUBSETS
        data_set_val = VideoDataset(args)
        data_loader_val = DataLoader(dataset=data_set_val,
                                      batch_size=args.BATCH_SIZE,
                                      num_workers=args.NUM_WORKERS,
                                      shuffle=True, pin_memory=True,
                                      collate_fn=custum_collate,
                                      drop_last=True)



        tasks_manager = TasksManager(data_loader_train=data_loader_train, data_loader_val=data_loader_val, seq_len=args.SEQ_LEN,
                                     labels_definition=data_set.get_labels_definition())
        tasks_manager.run_tasks_single()
        tasks_manager.run_multi_tasks()

    else:
        args.MAX_SEQ_STEP = 1
        args.SUBSETS = args.TEST_SUBSETS
        full_test = True  # args.MODE != 'train'

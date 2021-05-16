import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from model.dataLoader import VideoDataset
from model.datasets import custum_collate
from tasksManager import TasksManager
from utils import ioUtils as utils
import argparse

from utils.printUtility import print_info, print_warn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('DATA_ROOT', help='Location to root directory for dataset reading')  # /mnt/mars-fast/datasets/
    # parser.add_argument('SAVE_ROOT', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
    parser.add_argument('--DATASET', default='road',
                        type=str, help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_3,',
                        type=str, help='Training SUBSETS separated by ,')
    parser.add_argument('--VAL_SUBSETS', default='',
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

    args = parser.parse_args()
    args = utils.set_args(args)  # set SUBSETS of datasets

    if args.DEBUG:
        print_warn("-------------- Running in DEBUG mode --------------")

    if args.MODE in ['train', 'val']:
        print_info("Loading dataset ...")
        data_set = VideoDataset(args)
        print_info("Dataset loaded.")

        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.BATCH_SIZE,
                                 num_workers=args.NUM_WORKERS,
                                 shuffle=True, pin_memory=True,
                                 collate_fn=custum_collate,
                                 drop_last=True)

        tasks_manager = TasksManager(data_loader=data_loader, seq_len=args.SEQ_LEN)
        tasks_manager.run_tasks_single("ActiveAgentDetection")
        tasks_manager.run_multi_tasks()

    else:
        args.MAX_SEQ_STEP = 1
        args.SUBSETS = args.TEST_SUBSETS
        full_test = True  # args.MODE != 'train'
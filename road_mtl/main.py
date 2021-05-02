from torch.utils.data import DataLoader

from model.dataLoader import VideoDataset
from tasksManager import TasksManager
from utils import ioUtils as utils
import argparse

from utils.printUtility import print_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('DATA_ROOT', help='Location to root directory for dataset reading')  # /mnt/mars-fast/datasets/
    # parser.add_argument('SAVE_ROOT', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
    parser.add_argument('--DATASET', default='road',
                        type=str, help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_3,',
                        type=str, help='Training SUBSETS seprated by ,')
    parser.add_argument('--VAL_SUBSETS', default='',
                        type=str, help='Validation SUBSETS seprated by ,')
    parser.add_argument('--TEST_SUBSETS', default='',
                        type=str, help='Testing SUBSETS seprated by ,')
    parser.add_argument('--MODE', default='train',
                        help='MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes')
    parser.add_argument('--SEQ_LEN', default=4,
                        type=int, help='Number of input frames')
    parser.add_argument('-b', '--BATCH_SIZE', default=4,
                        type=int, help='Batch size for training')
    parser.add_argument('--MIN_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MAX_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MIN_SIZE', default=512,
                        type=int, help='Input Size for FPN')

    args = parser.parse_args()
    args = utils.set_args(args)  # set SUBSETS of datasets

    if args.MODE in ['train', 'val']:
        print_info("Loading dataset ...")
        data_set = VideoDataset(args)
        print_info("Dataset loaded.")

        data_loader = DataLoader(data_set, args.BATCH_SIZE,
                                 num_workers=args.NUM_WORKERS,
                                 shuffle=True, pin_memory=True,
                                 drop_last=True)

        tasks_manager = TasksManager(data_loader=data_loader, seq_len=args.SEQ_LEN)
        tasks_manager.run()

    else:
        args.MAX_SEQ_STEP = 1
        args.SUBSETS = args.TEST_SUBSETS
        full_test = True  # args.MODE != 'train'
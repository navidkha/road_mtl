

def set_args(args):
    args.MAX_SIZE = int(args.MIN_SIZE*1.35)

    args.TRAIN_SUBSETS = [val for val in args.TRAIN_SUBSETS.split(',') if len(val)>1]
    args.VAL_SUBSETS = [val for val in args.VAL_SUBSETS.split(',') if len(val)>1]
    args.TEST_SUBSETS = [val for val in args.TEST_SUBSETS.split(',') if len(val)>1]
    ## check if subsets are okay
    possible_subets = ['test', 'train','val']
    for idx in range(1,4):
        possible_subets.append('train_'+str(idx))        
        possible_subets.append('val_'+str(idx))        

    if len(args.VAL_SUBSETS) < 1 and args.DATASET == 'road':
        args.VAL_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
    if len(args.TEST_SUBSETS) < 1:
        args.TEST_SUBSETS = args.VAL_SUBSETS
    
    for subsets in [args.TRAIN_SUBSETS, args.VAL_SUBSETS, args.TEST_SUBSETS]:
        for subset in subsets:
            assert subset in possible_subets, 'subest should from one of these '+''.join(possible_subets)

    args.DATASET = args.DATASET.lower()

    args.MEANS =[0.485, 0.456, 0.406]
    args.STDS = [0.229, 0.224, 0.225]
    
    return args
from os import path as osp
import time
import argparse
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import download
from utils_training.utils import parse_list, log_args, boolean_string, flow2kps
from models import QueryMatching
from utils_training.evaluation import Evaluator

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Training Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='/home/dev4/data/SKY/datasets')
    parser.add_argument('--benchmark', type=str, default='spair', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--lr-backbone', type=float, default=3e-6, metavar='LR',
                        help='learning rate (default: 3e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[70, 80, 90]')
    parser.add_argument('--step_gamma', type=float, default=0.5)

    parser.add_argument('--feature-size', type=int, default=16)
    parser.add_argument('--feature-proj-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--hyperpixel', type=str, default='[0,8,20,21,26,28,29,30]')
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)

    # Queries
    parser.add_argument('--num_queries', type=int, default=32)
    parser.add_argument('--feat_dim', type=int, default=256)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    with open(osp.join(args.pretrained, 'args.pkl'), 'rb') as f:
        args_model = pickle.load(f)
    log_args(args_model)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False, args_model.feature_size)
    test_dataloader = DataLoader(test_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)

    # Model
    model = QueryMatching(freeze=True, num_queries=args.num_queries, feature_size=args.feature_size)

    if args.pretrained:
        checkpoint = torch.load(osp.join(args.pretrained, 'model_best.pth'))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError()
    # create summary writer

    model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()

    for mini_batch in test_dataloader :
        flow_gt = mini_batch['flow'].to(device)
        trg_img = mini_batch['trg_img'].to(device)
        src_img = mini_batch['src_img'].to(device)

        pred_flow = model(mini_batch['trg_img'].to(device),
                        mini_batch['src_img'].to(device))
        
        estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))


        break

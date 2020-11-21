import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
import os


from dataloader import *
from gbml.maml import MAML
from gbml.imaml import iMAML
from utils import set_seed, set_gpu, check_dir, dict2tsv, BestTracker

def train(args, model, dataloader):

    loss_list = []
    dice_list = []
    grad_list = []
    with tqdm(dataloader, total=args.num_train_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, dice_log, grad_log = model.outer_loop(batch, is_train=True)

            loss_list.append(loss_log)
            dice_list.append(dice_log)
            grad_list.append(grad_log)
            pbar.set_description('loss = {:.4f} || dice={:.4f} || grad={:.4f}'.format(np.mean(loss_list), np.mean(dice_list), np.mean(grad_list)))
            if batch_idx >= args.num_train_batches:
                break

    loss = np.round(np.mean(loss_list), 1)
    dice= np.round(np.mean(dice_list), 1)
    grad = np.round(np.mean(grad_list), 1)

    return loss, dice, grad

@torch.no_grad()
def valid(args, model, dataloader):

    loss_list = []
    dice_list = []
    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, dice_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            acc_list.append(dice_log)
            pbar.set_description('loss = {:.4f} || dice={:.4f}'.format(np.mean(loss_list), np.mean(dice_list)))
            if batch_idx >= args.num_valid_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    dice = np.round(np.mean(dice_list), 4)

    return loss, dice

@BestTracker
def run_epoch(epoch, args, model, train_loader, test_loader):

    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_dice, train_grad = train(args, model, train_loader)
    test_loss, test_dice = valid(args, model, test_loader)

    res['epoch'] = epoch
    res['train_loss'] = train_loss
    res['train_dice'] = train_dice
    res['train_grad'] = train_grad
    res['valid_loss'] = valid_loss
    res['valid_dice'] = valid_dice
    res['test_loss'] = test_loss
    res['test_dice'] = test_dice

    return res




def main(args):

    if args.alg=='MAML':
        model = MAML(args)

    elif args.alg=='iMAML':
        model = iMAML(args)
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    if args.load:
        model.load()
    elif args.load_encoder:
        model.load_encoder()

    transform=transforms.Compose([transforms.Resize((256,256)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.5,), (0.5,))
                                                         ])




    trainframe=pd.read_csv("/content/train_data.csv")
    testframe=pd.read_csv("/content/test_data.csv")
    train_classes=np.unique(trainframe["ID"])
    train_classes=list(train_classes)
    all_test_classes=np.unique(testframe["ID"])
    all_test_classes=list(all_test_classes)

    num_classes=args.num_way
    num_instances=args.num_shot

    num_test_classes=args.num_test_classes
    num_test_instances=args.num_shot





    train_fileroots_alltask,meta_fileroots_alltask =[],[]

    for each_task in range(args.num_task):
        task=Task(train_classes,num_classes,num_instances)
        train_fileroots_alltask.append(task.train_roots)
        meta_fileroots_alltask.append(task.meta_roots)


    test_fileroots_alltask,train_fileroots_all_task =[],[]

    for each_task in range(args.num_test_task):
        test_task= TestTask(all_test_classes,num_test_classes,num_instances,num_test_instances)
        test_fileroots_alltask.append(test_task.test_roots)
        train_fileroots_all_task.append(test_task.train_roots)



    trainloader=DataLoader(MiniSet(train_fileroots_alltask,transform=transform),
                                            batch_size=1,num_workers=4, pin_memory=True,shuffle=True)

    validloader = DataLoader(MiniSet(meta_fileroots_alltask,transform=transform),
                           batch_size=1, shuffle=True, num_workers=4,  pin_memory=True)


    meta_train_trainloader=DataLoader(MiniSet(train_fileroots_all_task,transform=transform),
                            batch_size=1,shuffle=True, num_workers=4,  pin_memory=True)


    testloader=DataLoader(MiniSet(test_fileroots_alltask,transform=transform),
                         batch_size=1,shuffle=True, num_workers=4,  pin_memory=True)


    for epoch in range(args.num_epoch):

        res, is_best = run_epoch(epoch, args, model,train_loader=zip(trainloader,validloader), test_loader=zip(meta_train_trainloader,testloader))
        dict2tsv(res, os.path.join(args.result_path, args.alg, args.log_path))

        if is_best:
            model.save()
        torch.cuda.empty_cache()

        if args.lr_sched:
            model.lr_sched()

    return None

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    # experimental settings
    parser.add_argument('--root_dir', type=str, default="./content")
    parser.add_argument('--seed', type=int, default=2020,
        help='Random seed.')
    parser.add_argument('--data_path', type=str, default='../data/',
        help='Path of MiniImagenet.')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='best_model.pth')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    # training settings
    parser.add_argument('--num_epoch', type=int, default=100,
        help='Number of epochs for meta train.')
    parser.add_argument('--batch_size', type=int, default=1,
        help='Number of tasks in a mini-batch of tasks (default: 1).')
    parser.add_argument('--num_train_batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 250).')
    parser.add_argument('--num_valid_batches', type=int, default=100,
        help='Number of batches the model is trained over (default: 150).')
    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=5,
        help='Number of support examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num_query', type=int, default=5,
        help='Number of query examples per class (k in "k-query", default: 5).')
    parser.add_argument('--num_way', type=int, default=3,
        help='Number of classes per task (N in "N-way", default: 3).')
    parser.add_argument('--num_test_classes', type=int, default=1,
            help='Number of classes in meta training testing set (N in "N-way", default: 1).')
    parser.add_argument('--alg', type=str, default='iMAML')

    parser.add_argument('--num_test_task', type=int, default=1,
        help='Number of test tasks ( default: 1).')
    parser.add_argument('--num_task', type=int, default=20,
        help='Number of  tasks ( default: 10).')
    # algorithm settings

    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-4)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-4)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    # network settings
    parser.add_argument('--net', type=str, default='ConvNet')
    parser.add_argument('--n_conv', type=int, default=4)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args)
    main(args)

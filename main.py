import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict



from dataloader import *
from Alg.imaml import iMAML
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
            pbar.set_description('loss = {:.4f} || tdice={:.4f} || grad={:.4f}'.format(np.mean(loss_list), np.mean(dice_list), np.mean(grad_list)))
            if batch_idx >= args.num_train_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    dice= np.round(np.mean(dice_list), 4)
    grad = np.round(np.mean(grad_list), 4)

    return loss, dice, grad

@torch.no_grad()
def valid(args, model, dataloader):

    loss_list = []
    dice_list = []
    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, dice_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            dice_list.append(dice_log)
            pbar.set_description('loss = {:.4f} || vdice={:.4f}'.format(np.mean(loss_list), np.mean(dice_list)))
            if batch_idx >= args.num_valid_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    dice = np.round(np.mean(dice_list),4)

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

    transform = A.Compose ([
    A.Resize(width = 256, height = 256, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate((-5,5),p=0.5),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
                                    num_flare_circles_lower=1, num_flare_circles_upper=2,
                                    src_radius=160, src_color=(255, 255, 255),  always_apply=False, p=0.3),
    A.RGBShift (r_shift_limit=10, g_shift_limit=10,
                  b_shift_limit=10, always_apply=False, p=0.2),
    A. ElasticTransform (alpha=2, sigma=15, alpha_affine=25, interpolation=1,
                                       border_mode=4, value=None, mask_value=None,
                                      always_apply=False,  approximate=False, p=0.3) ,
    A.Normalize( p=1.0),
    ToTensor(),
    ])


    target_set=args.target_set
    source_list = ['CVC-612', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']
    source_list.remove(target_set)

    root='/home/rabink1/Meta-Learning-Seg/polyp_sets'


    with open('train_data.csv','w') as csvfile:
         fields=['ID','Image_path','Label_path']
         csvwriter=csv.writer(csvfile,delimiter=',')
         csvwriter.writerow(fields)
         for idx,each_set in enumerate(source_list):
             for image in os.listdir(root+'/'+each_set+'/'+'masks') and os.listdir(root+'/'+each_set+'/'+'images'):
                 label_path=root+'/'+each_set+'/'+'masks'+'/'+image
                 image_path=root+'/'+each_set+'/'+'images'+'/'+image
                 csvwriter.writerow([idx,image_path,label_path])


    with open('test_data.csv','w') as csvfile:
         fields=['ID','Image_path','Label_path']
         csvwriter=csv.writer(csvfile,delimiter=',')
         csvwriter.writerow(fields)
         for image in os.listdir(root+'/'+target_set+'/'+'masks') and os.listdir(root+'/'+target_set+'/'+'images'):
             label_path=root+'/'+target_set+'/'+"masks"+'/'+image
             image_path=root+'/'+target_set+'/'+"images"+'/'+image
             csvwriter.writerow([target_set,image_path,label_path])

    trainframe=pd.read_csv("train_data.csv")
    testframe=pd.read_csv("test_data.csv")
    train_classes=np.unique(trainframe["ID"])
    train_classes=list(train_classes)
    all_test_classes=np.unique(testframe["ID"])
    all_test_classes=list(all_test_classes)

    num_classes=args.num_way
    num_instances=args.num_shot

    num_meta_testing_train=args.meta_testing_train_shots
    num_test_classes=args.num_test_classes
    num_meta_testing_test=args.meta_testing_test_shots





    train_fileroots_alltask,meta_fileroots_alltask =[],[]

    for each_task in range(args.num_task):
        task=Task(train_classes,num_classes,num_instances,trainframe)
        train_fileroots_alltask.append(task.train_roots)
        meta_fileroots_alltask.append(task.meta_roots)


    test_fileroots_alltask,train_fileroots_all_task =[],[]

    for each_task in range(args.num_test_task):
        test_task= TestTask(all_test_classes,num_test_classes,num_meta_testing_train,num_meta_testing_test,testframe)
        test_fileroots_alltask.append(test_task.test_roots)
        train_fileroots_all_task.append(test_task.train_roots)



    trainloader=DataLoader(MiniSet(train_fileroots_alltask,transform=transform),
                                            batch_size=args.batch_size,num_workers=4, pin_memory=True,shuffle=True)

    validloader = DataLoader(MiniSet(meta_fileroots_alltask,transform=transform),
                           batch_size=args.batch_size, shuffle=True, num_workers=4,  pin_memory=True)


    meta_train_trainloader=DataLoader(MiniSet(train_fileroots_all_task,transform=transform),
                            batch_size=args.batch_size,shuffle=True, num_workers=4,  pin_memory=True)


    testloader=DataLoader(MiniSet(test_fileroots_alltask,transform=transform),
                         batch_size=args.batch_size,shuffle=True, num_workers=4,  pin_memory=True)


    for epoch in range(args.num_epoch):
        torch.cuda.empty_cache()

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
    parser.add_argument('--target_set', type=str, default='CVC-612')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=True)

    parser.add_argument('--load_path', type=str, default="./weights/unet.pt")
    parser.add_argument('--device', type=int, nargs='+', default=[3], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')

    # training settings
    parser.add_argument('--num_epoch', type=int, default=70,
            help='Number of epochs for meta train.')

    parser.add_argument('--batch_size', type=int, default=1,
        help='Number of tasks in a mini-batch of tasks (default: 1).')

    parser.add_argument('--num_train_batches', type=int, default=250,
        help='Number of batches the model is trained over (default: 250).')

    parser.add_argument('--num_valid_batches', type=int, default=250,
        help='Number of batches the model is trained over (default: 150).')


    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=10,
        help='Number of support examples per class (k in "k-shot", default: 5).')

    parser.add_argument('--meta_testing_train_shots', type=int, default=30,
                   help='Number of meta testing train examples per class (k in "k-shot", default: 25).')

    parser.add_argument('--meta_testing_test_shots', type=int, default=30,
                   help='Number of meta testing test examples per class (k in "k-shot", default: 25).')


    parser.add_argument('--num_query', type=int, default=20,
        help='Number of query examples per class (k in "k-query", default: 5).')

    parser.add_argument('--num_way', type=int, default=3,
        help='Number of classes per task (N in "N-way", default: 3).')

    parser.add_argument('--num_test_classes', type=int, default=1,
            help='Number of classes in meta training testing set (N in "N-way", default: 1).')

    parser.add_argument('--alg', type=str, default='iMAML')

    parser.add_argument('--num_test_task', type=int, default=2,
        help='Number of test tasks ( default: 1).')

    parser.add_argument('--num_task', type=int, default=20,
        help='Number of  tasks ( default: 10).')


    # algorithm settings

    parser.add_argument('--n_inner', type=int, default=150)
    parser.add_argument('--inner_lr', type=float, default=1e-5)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-5)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args)
    main(args)

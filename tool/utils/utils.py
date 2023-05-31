import argparse
import os
from tool.utils.options import dict2str, parse
from tool.utils import set_random_seed
from tool.data import create_dataloader, create_dataset
from tool.data.data_sampler import EnlargedSampler
import random
import torch
import math
def parse_options(opt_path,is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default=opt_path, help='Path to option YAML file.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed )
    opt['dist'] = False
    opt['rank'], opt['world_size'] = 0,1
    return opt

def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])

        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def save_model(net, epoch, avg_loss,logdir):
    name = "epoch_{}_avg_loss_{:.2f}.pth".format(epoch,avg_loss)
    fname = os.path.join(logdir,name) 
    states = {
         'state_dict_net': net.state_dict(),
         'epoch': epoch,
    }
    torch.save(states, fname)
    torch.save(states, os.path.join(logdir,'best.pth'))
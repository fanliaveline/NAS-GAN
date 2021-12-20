# @Date    : 2019-10-22
# @Author  : Chen Gao

from __future__ import absolute_import, division, print_function


import cfg
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.genotype import alpha2genotype, beta2genotype, draw_graph_G, draw_graph_D

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from architect import Architect_gen, Architect_dis
from utils.flop_benchmark import print_FLOPs

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for _id in range(len(str_ids)):
        if _id >= 0:
            args.gpu_ids.append(_id)
    # if len(args.gpu_ids) > 1:
    #     args.gpu_ids = args.gpu_ids[1:]
    # else:
    #     args.gpu_ids = args.gpu_ids


    # import network 导入网络结构
    #basemodel_gen=archs.arch_cifar10.Generator
    basemodel_gen1= eval('archs.' + args.arch + '.Generator')(args=args)
    basemodel_gen2 = eval('archs.' + args.arch + '.Generator')(args=args)
    gen_net1 = torch.nn.DataParallel(basemodel_gen1,device_ids=args.gpu_ids,output_device=args.gpu_ids[0]).cuda(args.gpu_ids[0])
    gen_net2 = torch.nn.DataParallel(basemodel_gen2,device_ids=args.gpu_ids,output_device=args.gpu_ids[0]).cuda(args.gpu_ids[0])
#basemodel_dis=archs.arch_cifar10.Discriminator
    basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args=args)
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids,output_device=args.gpu_ids[0]).cuda(args.gpu_ids[0])

    architect_gen1 = Architect_gen(gen_net1, args)
    architect_gen2 = Architect_gen(gen_net2, args)
    architect_dis = Architect_dis(dis_net, args)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net1.apply(weights_init)
    gen_net2.apply(weights_init)
    dis_net.apply(weights_init)


    # set optimizer
    arch_params_gen1 = gen_net1.module.arch_parameters()
    arch_params_gen2 = gen_net2.module.arch_parameters()
    arch_params_gen_ids1 = list(map(id, arch_params_gen1))
    arch_params_gen_ids2= list(map(id, arch_params_gen2))
    weight_params_gen1 = filter(lambda p: id(p) not in arch_params_gen_ids1, gen_net1.parameters())
    weight_params_gen2 = filter(lambda p: id(p) not in arch_params_gen_ids2, gen_net2.parameters())
    gen_optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_gen1),
                                     args.g_lr, (args.beta1, args.beta2))
    gen_optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_gen2),
                                     args.g_lr, (args.beta1, args.beta2))

    arch_params_dis = dis_net.module.arch_parameters()
    arch_params_dis_ids = list(map(id, arch_params_dis))
    weight_params_dis = filter(lambda p: id(p) not in arch_params_dis_ids, dis_net.parameters())
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_dis),
                                     args.d_lr, (args.beta1, args.beta2))

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic  #5*5
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    args.max_iter_D = args.max_epoch_D * len(train_loader)

    gen_scheduler1 = LinearLrDecay(gen_optimizer1, args.g_lr, 0.0, 0, args.max_iter_D)
    gen_scheduler2 = LinearLrDecay(gen_optimizer2, args.g_lr, 0.0, 0, args.max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param1 = copy_params(gen_net1)
    gen_avg_param2 = copy_params(gen_net2)
    start_epoch = 0
    # best_fid = 1e4

    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    logger.info("GPU_ids", args.gpu_ids)
    logger.info("param size of G1 = %fMB", count_parameters_in_MB(gen_net1))
    logger.info("param size of G2 = %fMB", count_parameters_in_MB(gen_net2))
    logger.info("param size of D = %fMB", count_parameters_in_MB(dis_net))

    # search loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress'):
        lr_schedulers1 = (gen_scheduler1, dis_scheduler) if args.lr_decay else None
        lr_schedulers2 = (gen_scheduler2, dis_scheduler) if args.lr_decay else None
        tau_decay = np.log(args.tau_max / args.tau_min) / args.max_epoch_D if args.gumbel_softmax else None
        tau = max(0.1, args.tau_max * np.exp(-tau_decay * epoch)) if args.gumbel_softmax else None
        if tau:
            gen_net1.module.set_tau(tau)
            gen_net2.module.set_tau(tau)
            dis_net.module.set_tau(tau)

        # search arch and train weights
        if epoch > 0:
            train(args, gen_net1,gen_net2, dis_net, gen_optimizer1, gen_optimizer2, dis_optimizer, gen_avg_param1, gen_avg_param2, train_loader, epoch, writer_dict,
                  lr_schedulers1,lr_schedulers2, architect_gen1=architect_gen1, architect_gen2=architect_gen2,architect_dis=architect_dis)

        # save and visualise current searched arch
        if epoch == 0 or epoch % args.derive_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            genotype_G1 = alpha2genotype(gen_net1.module.alphas_normal, gen_net1.module.alphas_up, save=True,
                                        file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_G1.npy'))
            genotype_G2 = alpha2genotype(gen_net2.module.alphas_normal, gen_net2.module.alphas_up, save=True,
                                        file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_G2.npy'))
            genotype_D = beta2genotype(dis_net.module.alphas_normal, dis_net.module.alphas_down, save=True,
                                       file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_D.npy'))
            if args.draw_arch:
                draw_graph_G(genotype_G1, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_G1'))
                draw_graph_G(genotype_G2, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_G2'))
                draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_D'))

        # validate current searched arch
        if epoch == 0 or epoch % args.val_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            backup_param1 = copy_params(gen_net1)
            load_params(gen_net1, gen_avg_param1)
            backup_param2 = copy_params(gen_net2)
            load_params(gen_net2, gen_avg_param2)

            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net1,gen_net2, writer_dict)
            logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                        f'FID score: {fid_score} || @ epoch {epoch}.')

        avg_gen_net1 = deepcopy(gen_net1)
        load_params(avg_gen_net1, gen_avg_param1)
        avg_gen_net2 = deepcopy(gen_net2)
        load_params(avg_gen_net2, gen_avg_param2)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.arch,
            'gen_state_dict1': gen_net1.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict1': avg_gen_net1.state_dict(),
            'gen_optimizer1': gen_optimizer1.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'path_helper': args.path_helper
        }, False, args.path_helper['ckpt_path'],filename="checkpoint1"+str(epoch)+".pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.arch,
            'gen_state_dict2': gen_net2.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict2': avg_gen_net2.state_dict(),
            'gen_optimizer2': gen_optimizer2.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'path_helper': args.path_helper
        }, False, args.path_helper['ckpt_path'], filename="checkpoint2"+str(epoch)+".pth")
        del avg_gen_net1
        del avg_gen_net2


if __name__ == '__main__':
    main()

# @Date    : 2019-10-22
# @Author  : Chen Gao

import os
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import  sys
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from utils.genotype import alpha2genotype, beta2genotype, draw_graph_D, draw_graph_G
from auto_clip import auto_clip_with_sliding_window
from auto_clip import auto_clip_original


logger = logging.getLogger(__name__)


def train(args, gen_net1: nn.Module,gen_net2: nn.Module, dis_net: nn.Module, gen_optimizer1,gen_optimizer2, dis_optimizer, gen1_avg_param,gen2_avg_param, train_loader, epoch,
          writer_dict, lr_schedulers1,lr_schedulers2, architect_gen1=None,architect_gen2=None, architect_dis=None):
    writer = writer_dict['writer']
    gen_step = 0

    def Cal_Orthgonal(vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product = dot_product + a * b
            normA = normA + a ** 2
            normB = normB + b ** 2
        if normA == 0.0 or normB == 0.0:
            return 0.0
        else:
            return abs(dot_product) / (((normA * normB) ** 0.5)+1e-8)

    def multiple_Orthgonal(idx, output_G):
        MI = 0
        for i in range(2):
            if i == idx:
                continue
            else:
                G_output1 = output_G[idx]
                G_output2 = output_G[i]
                tmp = Cal_Orthgonal(G_output1, G_output2)
                MI = MI + tmp
        return MI

    # train mode
    gen_net1 = gen_net1.train()
    gen_net2 = gen_net2.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        real_imgs = imgs.type(torch.cuda.FloatTensor)
        real_imgs_w = real_imgs[:imgs.shape[0] // 2]
        real_imgs_arch = real_imgs[imgs.shape[0] // 2:]  #1-m用于优化结构，m-2m用于优化权重
        grad_history = [] 
        # sample noise
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0] // 2, args.latent_dim)))

        # search arch of D
        if architect_dis:
            # sample noise
            search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0] // 2, args.latent_dim)))
            if args.amending_coefficient:
                architect_dis.step(dis_net, real_imgs_arch, gen_net1,gen_net2, search_z, real_imgs_train=real_imgs_w, train_z=z, eta=args.amending_coefficient)
            else:
                architect_dis.step(dis_net, real_imgs_arch, gen_net1,gen_net2, search_z)
        # train weights of D
        doc=open("s-1.txt","a")
        print(global_steps,file=doc)
        dis_optimizer.zero_grad()
        real_validity , _ = dis_net(real_imgs_w)
        print("real_validity",real_validity,file=doc)
        #errD_real = torch.mean(torch.nn.ReLU(inplace=True)(1.0 - real_validity))
        #errD_real.backward()
        fake_imgs1 = gen_net1(z).detach()
        assert fake_imgs1.size() == real_imgs_w.size()
        fake_validity1, _ = dis_net(fake_imgs1)
        #writer.add_scalar('fake_val for D', fake_validity1, global_steps)
        #errD_fake1 = torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity1))
       # errD_fake1.backward()
        fake_imgs2 = gen_net2(z).detach()
        assert fake_imgs2.size() == real_imgs_w.size()
        fake_validity2, _ = dis_net(fake_imgs2)
        #errD_fake2 = torch.mean(torch.nn.ReLU(inplace=True)(1 + fake_validity2))
        #errD_fake2.backward()


        # Hinge loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 (torch.mean(nn.ReLU(inplace=True)(1 + fake_validity1))+torch.mean(nn.ReLU(inplace=True)(1 + fake_validity2)))/2.0
        d_loss.backward()
        #auto_clip_original(grad_history,dis_net)
        #auto_clip_with_sliding_window(grad_history,dis_net)
        dis_optimizer.step()

  
        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # sample noise
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
        # search arch of G
        if architect_gen1:# 判断是否是搜索阶段
            if global_steps % args.n_critic == 0:
                # sample noise
                search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
                if args.amending_coefficient:
                    architect_gen1.step(search_z, gen_net1, dis_net, train_z=gen_z, eta=args.amending_coefficient)
                else:
                    architect_gen1.step(search_z, gen_net1,dis_net)
        if architect_gen2:# 判断是否是搜索阶段
            if global_steps % args.n_critic == 0:
                # sample noise
                search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
                if args.amending_coefficient:
                    architect_gen2.step(search_z, gen_net2, dis_net, train_z=gen_z, eta=args.amending_coefficient)
                else:
                    architect_gen2.step(search_z, gen_net2,dis_net)

        # train weights of G
        if global_steps % args.n_critic == 0:
            gen_optimizer1.zero_grad()
            gen_imgs1 = gen_net1(gen_z)
            fake_validity1,errG1_l1 = dis_net(gen_imgs1)
            g_loss1 = -torch.mean(fake_validity1)


            print("fake_validity1 for G",fake_validity1,file=doc)
            print("ERRG1", errG1_l1,file=doc)
            gen_optimizer2.zero_grad()
            gen_imgs2 = gen_net2(gen_z)
            fake_validity2 ,errG2_l1= dis_net(gen_imgs2)
            g_loss2 = -torch.mean(fake_validity2)
            print("ERRG2", errG2_l1, file=doc)
            
            errG_MI_1 = g_loss1 / 2.0 + Cal_Orthgonal(errG1_l1, errG2_l1)
            print("g_loss1", g_loss1)
            print("errG_MI_1 ", errG_MI_1)
            errG_MI_1.backward(retain_graph=True)
            gen_optimizer1.step()

            errG_MI_2 = g_loss2 / 2.0 + Cal_Orthgonal(errG1_l1, errG2_l1)
            errG_MI_2.backward()
            gen_optimizer2.step()

            # Hinge loss（SNGAN loss）
            avg_fake_validity=(fake_validity1+fake_validity2)/2.0

            # g_loss1.backward(retain_graph=True)
            # gen_optimizer1.step()

            # g_loss2.backward(retain_graph=True)
            # gen_optimizer2.step()
           #  Tensor = torch.FloatTensor
           #  output_G = Tensor(args.gen_bs, args.gen_bs * 2)
           #  output_G[0] = errG1_l1
           #  output_G[1] = errG2_l1



            # learning rate
            if lr_schedulers1 and lr_schedulers2:
                gen_scheduler1, dis_scheduler = lr_schedulers1
                gen_scheduler2, _ = lr_schedulers2
                g_lr1 = gen_scheduler1.step(global_steps)
                g_lr2 = gen_scheduler2.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr1', g_lr1, global_steps)
                writer.add_scalar('LR/g_lr2', g_lr2, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p1, avg_p1 in zip(gen_net1.parameters(), gen1_avg_param):
                avg_p1.mul_(0.999).add_(0.001, p1.data)
            for p2, avg_p2 in zip(gen_net2.parameters(), gen2_avg_param):
                avg_p2.mul_(0.999).add_(0.001, p2.data)

            #g_loss=(errG_MI_1+errG_MI_2)/2.0
            # writer.add_scalar('g1_loss', g_loss1.item(), global_steps)
            # writer.add_scalar('g2_loss', g_loss2.item(), global_steps)
            writer.add_scalar('g1_loss', errG_MI_1.item(), global_steps)
            writer.add_scalar('g2_loss', errG_MI_2.item(), global_steps)

            gen_step += 1
        # if gen_step and iter_idx % args.print_freq == 0:
        #     tqdm.write(
        #         '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G1 loss: %f] [G2 loss: %f]' %
        #         (epoch, args.max_epoch_D, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss1.item(),g_loss2.item()))

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G1 loss: %f] [G2 loss: %f]' %
                (epoch, args.max_epoch_D, iter_idx % len(train_loader), len(train_loader), d_loss.item(), errG_MI_1.item(),errG_MI_2.item()))
        writer_dict['train_global_steps'] = global_steps + 1

        if architect_gen1 and architect_gen2 :
            # deriving arch of G/D during searching
            derive_freq_iter = math.floor((args.max_iter_D / args.max_epoch_D) / args.derive_per_epoch)
            if (args.derive_per_epoch > 0) and (iter_idx % derive_freq_iter == 0):
                genotype_G1 = alpha2genotype(gen_net1.module.alphas_normal, gen_net1.module.alphas_up, save=True,
                                            file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch)+'_'+str(iter_idx)+'_G1.npy'))
                genotype_G2 = alpha2genotype(gen_net2.module.alphas_normal, gen_net2.module.alphas_up, save=True,
                                            file_path=os.path.join(args.path_helper['genotypes_path'],
                                                                   str(epoch) + '_' + str(iter_idx) + '_G2.npy'))
                genotype_D = beta2genotype(dis_net.module.alphas_normal, dis_net.module.alphas_down, save=True,
                                           file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch)+'_'+str(iter_idx)+'_D.npy'))
                if args.draw_arch:
                    draw_graph_G(genotype_G1, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch)+'_'+str(iter_idx)+'_G1'))
                    draw_graph_G(genotype_G2, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'],
                                                        str(epoch) + '_' + str(iter_idx) + '_G2'))
                    draw_graph_D(genotype_D, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch)+'_'+str(iter_idx)+'_D'))


def validate(args, fixed_z, fid_stat, gen_net1: nn.Module,gen_net2: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net1 = gen_net1.eval()
    gen_net2 = gen_net2.eval()

    # generate images
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    sample_imgs1 = gen_net1(fixed_z)
    sample_imgs2 = gen_net2(fixed_z)
    sample_imgs=torch.cat([sample_imgs1.detach(),sample_imgs2.detach()],0)
    img_grid = make_grid(sample_imgs, nrow=10, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # generate a batch of images
        with torch.no_grad():
            gen_imgs1= gen_net1(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            gen_imgs2 = gen_net2(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',torch.uint8).numpy()
        for img_idx1, img1 in enumerate(gen_imgs1):
            file_name1 = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx1}_1.png')
            imsave(file_name1, img1)
        for img_idx2, img2 in enumerate(gen_imgs2):
            file_name2 = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx2}_2.png')
            imsave(file_name2, img2)
        img_list.extend(list(gen_imgs1))
        img_list.extend(list(gen_imgs2))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    
    # del buffer
    os.system('rm -r {}'.format(fid_buffer_dir))
    
    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, std, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

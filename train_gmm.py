import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from models import networks, dataset
writer = SummaryWriter('runs/gmm')

class Args:
    batchSize = 8
    dataroot = '../../DeepFashion_Try_On/acgpn_dataset/'
    datapairs = 'train_pairs.txt'
    phase = 'train'
    beta1 = 0.5

class Args2:
    batchSize = 2
    dataroot = '../../DeepFashion_Try_On/acgpn_dataset/'
    datapairs = 'val_pairs.txt'
    phase = 'val'
    
opt_train = Args
opt_val = Args2

gmm = networks.GMM(7, 3)
gmm.cuda()

discriminator = networks.Discriminator(7).cuda()

train_dataset = dataset.BaseDataset(opt_train)
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt_train.batchSize,
            shuffle=False)

val_dataset = dataset.BaseDataset(opt_val)
val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt_val.batchSize,
            shuffle=False)

def discriminate(netD ,input_label, real_or_fake):
    input = torch.cat([input_label, real_or_fake], dim=1)
    return netD.forward(input)

sigmoid = nn.Sigmoid()
tanh = torch.nn.Tanh()
l1loss = nn.L1Loss()
optimizerG = optim.AdamW(gmm.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))
optimizerD = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(opt_train.beta1, 0.999))
step = 0
step_val = 0
criterionVGG = networks.VGGLoss()
criterionGAN = networks.GANLoss(use_lsgan=False, tensor=torch.cuda.FloatTensor)
criterionFeat = nn.L1Loss()

if not os.path.isdir('checkpoint_gmm'):
    os.mkdir('checkpoint_gmm')
discriminator.train()
for epoch in range(20):
    gmm.train()
    for data in train_dataloader:
        t_mask=torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float))
        data['label']=data['label']*(1-t_mask)+t_mask*4
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        in_mask_clothes = Variable(mask_clothes.cuda())
        in_image = Variable(data['image'].cuda())
        in_edge = Variable(data['edge'])
        in_color = Variable(data['color'].cuda())
        in_skeleton = Variable(data['skeleton'].cuda())
        pre_clothes_mask = torch.FloatTensor((in_edge.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes = in_color*pre_clothes_mask
        fake_c, affine = gmm(clothes, in_mask_clothes, in_skeleton)
        fake_c=tanh(fake_c)

        input_pool = torch.cat([in_mask_clothes,clothes],1)
        real_pool = in_image*in_mask_clothes
        fake_pool = fake_c
        D_pool = discriminator

        loss_D_fake=0
        loss_D_real=0
        loss_G_VGG = 0

        pred_fake = discriminate(D_pool, input_pool.detach(), fake_pool.detach())
        loss_D_fake += criterionGAN(pred_fake, False)
        pred_real = discriminate(D_pool, input_pool.detach(), real_pool.detach())
        loss_D_real = criterionGAN(pred_real, True)
        pred_fake = D_pool.forward(torch.cat((input_pool.detach(), fake_pool.detach()), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)

        loss_G_VGG += criterionVGG(fake_pool, real_pool) *20
        L1_loss = criterionFeat(fake_pool, real_pool)
        L1_loss += criterionFeat(affine, real_pool)
        loss_D = loss_D_fake + loss_D_real * 0.5
        loss_G = loss_G_GAN + loss_G_VGG + L1_loss

        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        if epoch > 4:
            optimizerD.zero_grad()
            loss_D.backward()
            optimizerD.step()

        if step % 300 == 0:
            writer.add_image('torso_label', fake_c[0], step)
            writer.add_image('affine', affine[0], step)
            writer.add_image('gt', (in_image*in_mask_clothes)[0], step)
        if step % 5000 == 0:
            torch.save({
                'sgm': gmm.state_dict(),
                'optimizer_sgm': optimizerG.state_dict(),
                'disc': discriminator.state_dict(),
                'optimizer_disc': optimizerD.state_dict()
                }, 'checkpoint_gmm/model_'+str(step)+'.pth')

        step += 1
        writer.add_scalar('generator loss', loss_G, step)
        writer.add_scalar('discriminator loss', loss_D, step)
    gmm.eval()
    for data in val_dataloader:
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        in_mask_clothes = Variable(mask_clothes.cuda())
        in_image = Variable(data['image'].cuda())
        in_edge = Variable(data['edge'])
        in_color = Variable(data['color'].cuda())
        in_skeleton = Variable(data['skeleton'].cuda())
        pre_clothes_mask = torch.FloatTensor((in_edge.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes = in_color * pre_clothes_mask
        fake_c, affine = gmm(clothes, in_mask_clothes, in_skeleton)
        fake_c=tanh(fake_c)
        loss_G = l1loss(fake_c, in_image*in_mask_clothes)

        writer.add_scalar('loss_g_val', loss_G, step_val)
        step_val += 1

    torch.save(gmm.state_dict(), "checkpoint_gmm/gmm_"+str(epoch)+".pth")
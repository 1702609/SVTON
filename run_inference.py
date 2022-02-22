import torch
import torch.nn as nn
import os
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import cv2
from models import networks
from torch.autograd import Variable

class Args:
    batchSize = 1
    dataroot = '../../DeepFashion_Try_On/acgpn_dataset'
    datapairs = 'short_long.txt'
    #dataroot = 'data'
    #datapairs = 'test_pairs.txt'
    phase = 'test'
opt = Args

def get_transform(normalize=True):
        transform_list = []
        transform_list += [transforms.ToTensor()]
        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
    
class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        super(BaseDataset, self).__init__()
        
        human_names = []
        cloth_names = []
        with open(os.path.join(opt.dataroot, opt.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                human_names.append(h_name)
                cloth_names.append(c_name)
        self.human_names = human_names
        self.cloth_names = cloth_names

    def __getitem__(self, index):        
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]
        A_path = osp.join(self.opt.dataroot, self.opt.phase ,self.opt.phase + '_label', h_name.replace(".jpg", ".png"))
        label = Image.open(A_path).convert('L')

        B_path = osp.join(self.opt.dataroot, self.opt.phase ,self.opt.phase + '_img', h_name)
        image = Image.open(B_path).convert('RGB') 

        E_path = osp.join(self.opt.dataroot, self.opt.phase ,self.opt.phase + '_edge', c_name)
        edge = Image.open(E_path).convert('L')
                
        C_path = osp.join(self.opt.dataroot, self.opt.phase ,self.opt.phase + '_color', c_name)
        color = Image.open(C_path).convert('RGB')
        
        S_path = osp.join(self.opt.dataroot, self.opt.phase ,self.opt.phase + '_posergb', h_name)
        skeleton = Image.open(S_path).convert('RGB')

        M_path = osp.join(self.opt.dataroot, self.opt.phase ,self.opt.phase + '_imgmask', h_name.replace('.jpg', '.png'))
        mask = Image.open(M_path).convert('L')
        mask_array = np.array(mask)
        parse_shape = (mask_array > 0).astype(np.float32)
        parse_shape_ori = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape_ori.resize(
            (192 // 16, 256 // 16), Image.BILINEAR)
        mask = parse_shape.resize(
            (192, 256), Image.BILINEAR)

        transform_A = get_transform(method=Image.NEAREST, normalize=False)
        label_tensor = transform_A(label) * 255
        transform_B = get_transform()      
        image_tensor = transform_B(image)
        edge_tensor = transform_A(edge)
        color_tensor = transform_B(color)
        skeleton_tensor = transform_B(skeleton)
        mask_tensor = transform_A(mask)
        normal_tensor = transform_A(parse_shape_ori)
        return {'label': label_tensor, 'image': image_tensor,
                             'edge': edge_tensor,'color': color_tensor, 
                             'mask': mask_tensor, 'name' : c_name,
                             'colormask': mask_tensor, 'skeleton': skeleton_tensor,
                             #'pose':pose_map,
                             'blurry': mask_tensor, 'normal': normal_tensor}
    def __len__(self):
        return len(self.human_names)

t = BaseDataset(opt)
dataloader = torch.utils.data.DataLoader(
            t,
            batch_size=opt.batchSize,
            shuffle=False)

position = -1 
for data in dataloader:
    position +=1
    if position == 1:
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = data['image'] * mask_fore
        in_label = Variable(data['label'].cuda())
        in_edge = Variable(data['edge'].cuda())
        in_img_fore = Variable(img_fore.cuda())
        in_mask_clothes = Variable(mask_clothes.cuda())
        in_color = Variable(data['color'].cuda())
        in_image = Variable(data['image'].cuda())
        in_mask_fore = Variable(mask_fore.cuda()) 
        in_skeleton = Variable(data['skeleton'].cuda())
        in_blurry = Variable(data['blurry'].cuda())
        break

with torch.no_grad():
    phpm = networks.PHPM(7, 4)
    phpm.cuda()
    phpm.load_state_dict(torch.load('checkpoint/PHPM.pth'))
    phpm.eval()

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise.cuda()

def encode_input(label_map, clothes_mask,all_clothes_label):
    size = label_map.size()
    oneHot_size = (size[0], 14, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    masked_label= torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    masked_label=masked_label.scatter_(1,(label_map*(1-clothes_mask)).data.long().cuda(), 1.0)

    c_label=torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    c_label=c_label.scatter_(1,all_clothes_label.data.long().cuda(),1.0)

    input_label = Variable(input_label)

    return input_label,masked_label,c_label

pre_clothes_mask = torch.FloatTensor((in_edge.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
clothes = in_color*pre_clothes_mask
shape = pre_clothes_mask.shape

sigmoid = nn.Sigmoid()
phpm_in = torch.cat([in_blurry, clothes, in_skeleton], dim=1)
arm_label = phpm(phpm_in)
arm_label = sigmoid(arm_label)

def generate_discrete_label(inputs, label_nc, onehot=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

armlabel_map = generate_discrete_label(arm_label.detach(), 4, False)
dis_label = generate_discrete_label(arm_label.detach(), 4)
fake_cl = torch.FloatTensor((armlabel_map.cpu().numpy() == 1).astype(np.float)).cuda()

with torch.no_grad():
    G3 = networks.TOM(11, 3)
    G3.cuda(opt.gpu_ids[0])
    G3.load_state_dict(torch.load('checkpoint/TOM.pth')) 
    G3.eval()

with torch.no_grad():
    gmm = networks.GMM(7,3)
    gmm.cuda(opt.gpu_ids[0])
    gmm.load_state_dict(torch.load('checkpoint/GMM.pth'))
    gmm.eval()

def encode(label_map, size):
    label_nc = 14
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label

def ger_average_color(mask, arms):
    color = torch.zeros(arms.shape).cuda()
    for i in range(arms.shape[0]):
        count = len(torch.nonzero(mask[i, :, :, :]))
        if count < 10:
            color[i, 0, :, :] = 0
            color[i, 1, :, :] = 0
            color[i, 2, :, :] = 0

        else:
            color[i, 0, :, :] = arms[i, 0, :, :].sum() / count
            color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
            color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
    return color

arm1_mask = torch.FloatTensor((in_label.cpu().numpy() == 11).astype(np.float)).cuda()
arm2_mask = torch.FloatTensor((in_label.cpu().numpy() == 13).astype(np.float)).cuda()
skin_color = ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
            (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * in_image)
new_arm1_mask = torch.FloatTensor(
            (armlabel_map.cpu().numpy() == 2).astype(np.float)).cuda()
new_arm2_mask = torch.FloatTensor(
            (armlabel_map.cpu().numpy() == 3).astype(np.float)).cuda()

arm1_occ = in_mask_clothes * new_arm1_mask
arm2_occ = in_mask_clothes * new_arm2_mask
bigger_arm1_occ = arm1_occ
bigger_arm2_occ = arm2_occ

occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask+in_mask_clothes)) * \
          (1 - bigger_arm2_occ * (arm2_mask + arm1_mask+in_mask_clothes))
img_hole_hand = in_img_fore * (1 - in_mask_clothes) * occlude * (1 - fake_cl)
dis_label = encode(armlabel_map, armlabel_map.shape)


# In[29]:


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

tanh = torch.nn.Tanh()

fake_c, warped = gmm(clothes, fake_cl, in_skeleton)
fake_c=tanh(fake_c)

for image_index in range(img_hole_hand.shape[0]): # Produces G (handless segmentation)
    preserved = img_hole_hand[image_index].permute(1,2,0).detach().cpu().numpy()
    generate_map = armlabel_map[image_index].permute(1,2,0).detach().cpu().numpy()
    size = generate_map.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if preserved[i][j][0] > 0:
                generate_map[i][j] = 0
transform_A = get_transform(method=Image.NEAREST, normalize=False)
generate_map = transform_A(generate_map).cuda()
generate_map=generate_map.unsqueeze(0)

G3_in = torch.cat([img_hole_hand, generate_map, fake_c, skin_color, gen_noise(shape)], 1)
fake_image = G3(G3_in.detach())
fake_image = tanh(fake_image)

tensor = (fake_image[0].clone() + 1) * 0.5 * 255
tensor = tensor.cpu().clamp(0, 255)
array = tensor.detach().numpy().astype('uint8')
array = array.swapaxes(0, 1).swapaxes(1, 2)
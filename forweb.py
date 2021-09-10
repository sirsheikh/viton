import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image 

def transformImg(img):
	edgeTensor= transforms.ToTensor()
	edgeImgtensor=edgeTensor(img)
	return edgeImgtensor

def main():
	opt = TestOptions().parse()
	start_epoch, epoch_iter = 1, 0
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)

	warp_model = AFWM(opt, 3)
	#print(warp_model)
	warp_model.eval()
	warp_model.cuda()
	load_checkpoint(warp_model, opt.warp_checkpoint)

	gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
	#print(gen_model)
	gen_model.eval()
	gen_model.cuda()
	load_checkpoint(gen_model, opt.gen_checkpoint)

	#transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

	real_image = Image.open("dataset/test_img/img.jpg").convert('RGB')
	clothes = Image.open("dataset/test_clothes/img.jpg").convert('RGB')
	real_image=transformImg(real_image)
	clothes=transformImg(clothes)
	edge = Image.open(r"dataset/test_edge/img.jpg").convert('L')
	print(type(edge))
	print(edge)
	edge = transformImg(edge)
	edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
	#print(real_image)
	clothes = clothes * edge

	flow_out = warp_model(real_image.cuda(), clothes.cuda())
	'''
	warped_cloth, last_flow, = flow_out
	warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
	gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
	gen_outputs = gen_model(gen_inputs)
	p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
	p_rendered = torch.tanh(p_rendered)
	m_composite = torch.sigmoid(m_composite)
	m_composite = m_composite * warped_edge
	p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

	a = real_image.float().cuda()
	b= clothes.cuda()
	c = p_tryon
	combine = torch.cat([c[0]], 2).squeeze()
	cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
	rgb=(cv_img*255).astype(np.uint8)
	bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('results/demo/PFAFN/gh.jpg',bgr)


'''




if __name__ == '__main__':
	main()
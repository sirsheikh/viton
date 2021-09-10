from django.shortcuts import render
from django.http import HttpResponse
from subprocess import run,PIPE
from rip.forms import HotelForm
from django.conf import settings
from PIL import Image
import sys
import os
import cv2
import numpy as np
#import base64

# Create your views here.
def home_rip(request,*args,**kwargs):
	return render(request,"index.html") 

def result(request):
	if request.method == 'POST':
		img='ok'
		dressImg='ok'
		img=request.FILES.get('myPhoto')
		imgPIL=Image.open(img)
		imgPIL.save('dataset/test_img/edgemen.jpg')
		dressImg=request.POST['dress']
		edgeImgName=dressImg
		dressImg=Image.open(dressImg[1:])

		#nam=os.path.join('dataset/test_edge/',edgeImgName[15:])
		#print(nam)
		
		edgeimg=Image.open(os.path.join('dataset/test_edge/',edgeImgName[15:]))
		dressImg.save('dataset/test_clothes/dress.jpg')
		edgeimg.save('dataset/test_edge/dressEdge.jpg')
		
		a=run([sys.executable,"test.py"],shell=False,stdout=PIPE)
	return HttpResponse('ok')
		


  

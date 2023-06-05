import pydicom as dicom
from glob import glob
import cv2
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as ttf
from degradation import random_mixed_kernels, random_add_gaussian_noise, random_add_poisson_noise, random_add_jpg_compression, circular_lowpass_kernel

device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_degradation(img, semi=False):
  if not semi:
    kernel_list = ['plateau_iso','plateau_aniso']
    kernel_prob=[0.7,0.3]
    kernel=random_mixed_kernels(kernel_list, kernel_prob, kernel_size=21)
    img = cv2.filter2D(src=img, kernel=kernel, ddepth=-1)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  interpolation_list = [cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LINEAR]
  interpolation_prob = [0.6,0.2,0.2]
  interpolation = random.choices(interpolation_list, interpolation_prob)[0]
  size=img.shape[0]
  img = cv2.resize(img, (size//2, size//2), interpolation = interpolation)
  if not semi:
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    noise_list = ['gaussian', 'poisson']
    noise_prob = [0.5,0.5]
    noise = random.choices(noise_list, noise_prob)[0]
    if noise == 'gaussian':
      img = random_add_gaussian_noise(img)
    else :
      img = random_add_poisson_noise(img)
  img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  img = random_add_jpg_compression(img)
  img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  return img

def get_LR_image(img):
  img = apply_degradation(img)
  img = apply_degradation(img, True)
  sinc_kernel = circular_lowpass_kernel(cutoff=np.random.uniform(np.pi / 3, np.pi), kernel_size=11)
  img = cv2.filter2D(src=img, kernel=sinc_kernel, ddepth=-1)
  img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  img[img<=0]=0
  return img


class MyDataSet(Dataset):
    # Data Building
    def __init__(self,inputs,transform=None):
        super().__init__()

        self.x = inputs
        
        self.n_samples = len(inputs)
        self.transform = transform
    
    # get an Item
    def __getitem__(self,index):
        inputs = self.x[index]
        hr = self.transform(inputs) 
        lr = get_LR_image(inputs)
        lr = self.transform(lr)
        return hr, lr
    
    def __len__(self):
        return self.n_samples

def read_dicom(image_path):
  print (f'Total of {len(image_path)} DICOM images.' )
  slices = [dicom.read_file(path) for path in image_path]
  images=[]
  for ct in slices:
      img=ct.pixel_array
      norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      images.append(norm_img)
  return images

def create_dataset(root_path, batch_size=1):
  CT_path=[]
  for path in os.listdir(root_path):
    folders = os.listdir(root_path+path)
    for f in folders:
      subdir = os.listdir(root_path + path + '/' + f)
      for impath in subdir:
        image_path = root_path + path + '/' + f + '/' + impath
        data_paths = glob(image_path + '/*.dcm')
        data_paths.sort()
        CT_path += data_paths

  CT_images=read_dicom(CT_path)

  transform_hr = ttf.ToTensor()
  data = MyDataSet(CT_images,transform_hr)
  print('Loading data ... ')
  rn_data=[]
  for i in range(data.n_samples):
    rn_data.append(data.__getitem__(i))
  return rn_data

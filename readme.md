{\rtf1\ansi\ansicpg1252\cocoartf2706
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;\red0\green0\blue0;\red24\green26\blue30;
\red244\green246\blue249;}
{\*\expandedcolortbl;;\cssrgb\c100000\c100000\c100000;\cssrgb\c0\c0\c0;\cssrgb\c12157\c13725\c15686;
\cssrgb\c96471\c97255\c98039;}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh16380\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
The repository contains the implementation of the following paper.\
Title - BliMSR: Blind degradation modelling for generating high-resolution medical images\
Authors - \cb2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Samiran Dey, Partha Basuchowdhuri, Debasis Mitra, Robin Augustine\strokec0 , \strokec3 Sanjoy Kumar Saha\strokec0  \strokec3 and Tapabrata Chakraborti\strokec0 \
DOI - \
\
\
# Getting started\
\
##Installation\
To install all requirements execute the following line.\
```bash\
pip install -r requirements.txt
\f1 \cf4 \cb5 \strokec4 \

\f0 \cf0 \cb2 \strokec0 ``\
\
## Dataset Preparation\
The file dataset_random.py helps in preparing the data for training and testing. \
\
1. Prepare the data tensors as follows -\
```bash\
from dataset_random import create_dataset\
path = HR_image_path\
data = create_dataset(path)\
```\
\
2. Save the data tensors for further processing - \
```bash\
import torch\
torch.save(data, save_path)\
```\
\
## Train the model\
In trainer.py set paths for checkpoints and data. To begin training execute the following command.\
```bash\
python3 trainer.py\
```\
\
# Acknowledgements \
MedSRGAN - https://github.com/04RR/MedSRGAN\
Real_ESRGAN - https://github.com/xinntao/Real-ESRGAN/tree/5ca1078535923d485892caee7d7804380bfc87fd\
MHCA - https://github.com/lilygeorgescu/MHCA\
\
#Citation\
```bash\
\
```\
}
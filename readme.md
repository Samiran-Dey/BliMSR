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


from generator import Generator
from discriminator import DiscriminatorSN
from feature_extractor import vgg19

from tqdm import tqdm
import os
import torch
import time
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity

checkpoints_dir = 'Code/Blind_MedSRGAN/checkpoints_random/' #path for checkpoint
data_path = 'Dataset/LIDC-IDRI/train_data.pt' #path to LR, HR image pair
batch_size=4
start_epoch = 1
epochs=150
device = "cuda:0" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
if not os.path.exists(checkpoints_dir):
  os.makedirs(checkpoints_dir)

gen = Generator().to(device)
disc = DiscriminatorSN().to(device)
feature_extractor = vgg19().to(device)
feature_extractor.eval()

data = torch.load(data_path)
train_dl = DataLoader(data, batch_size=batch_size, shuffle=False)

optimizer_G = optim.Adam(gen.parameters(), lr=1e-5)
optimizer_D = optim.Adam(disc.parameters(), lr=1e-5)
loss_function = torch.nn.L1Loss().to(device)
gan_loss = torch.nn.BCEWithLogitsLoss().to(device)
scaler = torch.cuda.amp.GradScaler()

#function to compute gradient loss
def gradient_loss(preds, targets):
  gr_loss=0
  for pred, target in zip(preds,targets):
    image1=pred[0].detach().cpu().numpy()
    image2=target[0].detach().cpu().numpy()
    gr1=np.array(np.gradient(image1))
    gr2=np.array(np.gradient(image2))
    gr_loss+=np.mean(abs(gr1-gr2))
  return gr_loss/len(preds)

#function to compute structural dissimilarity loss
def ssim_loss(preds, targets):
    transform = T.ToPILImage()
    sm_loss=0
    for pred, target in zip(preds,targets):   	
      image1= np.asarray(transform(pred[0]))
      image2= np.asarray(transform(target[0]))
      sm = structural_similarity(image1, image2)
      sm_loss += (1-abs(sm))/2
    return sm_loss/len(preds)

#function to compute perceptual loss
def perceptual_loss(preds, target):
  sr = torch.cat((preds,preds,preds),dim=1)
  hr = torch.cat((target,target,target),dim=1)
  pred_features = feature_extractor(sr)
  hr_features = feature_extractor(hr)

  feature_loss = 0.0
  w=(0.5,.25,.125,0.0625,0.0625)
  i=0
  for pred_feature, hr_feature in zip(pred_features, hr_features):
      feature_loss += w[i]*loss_function(pred_feature, hr_feature)
      i+=1
  return feature_loss

log_name = os.path.join(checkpoints_dir, 'loss_log.txt')
with open(log_name, "a") as log_file:
    now = time.strftime("%c")
    log_file.write('================ Training Loss (%s) ================\n' % now)

t_loss_G, t_loss_D = [], []

for epoch in range(start_epoch,epochs+1):
    e_loss_G, e_loss_D = [], []

    for data in tqdm(train_dl):
        hr_img, lr_img = data
        hr_img=hr_img.to(device)
        lr_img=lr_img.to(device)

        valid = Variable(Tensor(np.ones((hr_img.shape[0], 1))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((hr_img.shape[0], 1))), requires_grad=False)
        transform=T.Resize((hr_img.shape[2], hr_img.shape[3]),interpolation=T.InterpolationMode.NEAREST)
        with torch.cuda.amp.autocast():

            # Train Generator

            pred_hr = gen(lr_img)

            content_loss = loss_function(pred_hr, hr_img)
            dssim_loss = ssim_loss(pred_hr,hr_img)
            gr_loss = gradient_loss(pred_hr, hr_img)
            perc_loss = perceptual_loss(pred_hr,hr_img)
            adv_loss = gan_loss(disc(pred_hr),valid)
            
            #loss function for generator
            loss_G = perc_loss + 0.01*adv_loss + 0.1*content_loss + 1*gr_loss + 1*dssim_loss

            optimizer_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()
            e_loss_G.append(float(loss_G))

            #Train Discriminator
            pred_real = disc(hr_img)
            pred_fake = disc(pred_hr.detach())
            
            #loss function for discriminator
            loss_D = (gan_loss(pred_real,valid)+gan_loss(pred_fake,fake))/2

            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()
            e_loss_D.append(float(loss_D))
            msg='gan=%.5f adv=%.5f content=%.5f gr=%.5f perc=%.5f dssim=%.5f dis=%.5f'% (loss_G,adv_loss,content_loss,gr_loss,perc_loss,dssim_loss,loss_D)
            with open(log_name, "a") as log_file:
              log_file.write('%s\n' % msg)  # save the message

    t_loss_D=sum(e_loss_D) / len(e_loss_D)
    t_loss_G=sum(e_loss_G) / len(e_loss_G)

    message = f"{epoch}/{epochs} -- Gen Loss: {t_loss_G} -- Disc Loss: {t_loss_D}"
    print(message)

    with open(log_name, "a") as log_file:
      log_file.write('%s\n' % message)  # save the message

    if epoch%10==0 or epoch==epochs:
      torch.save(gen.state_dict(), checkpoints_dir+"gen_"+str(epoch)+'.pth')
      torch.save(disc.state_dict(), checkpoints_dir+"disc_"+str(epoch)+'.pth')

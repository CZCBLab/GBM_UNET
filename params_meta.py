import os
import sys
import torch
from termcolor import colored
sys.path.append(os.path.join(sys.path[0], 'segmentation/unet'))
from unet.unet_loss import IoULoss, FocalLoss, DiceLoss, TverskyLoss

# ============================== parameters ==============================
# global parameters
env = 'local'
DEVICE = {
  'server': 'cuda',
  'local': 'cuda'
}

# pre-processing parameters
datasets = ['ILK', '4wks']
kfold = 4 # 0 if all in inputs, 1 if all in tests, 5 by default (80% train 20% test)
sliding_step = 300
PATH = {
  # 'server': '/projectnb/czlab/A08_Lu_image/Week_2-6-2023/copy/',
  # 'local': '/Users/ericwang/Desktop/Research/Digital-Biopsy/'
  'server': '.\\projectnb\\czlab\\A08_Lu_image\\Week_2-6-2023\\copy\\',
  'local': '../data/digital-biopsy/dataset'
}
DATASET = {
  # '4wks': 'train-data-4wks/',
  # '16wks': 'train-data-16wks/',
  # 'ILK': 'train-data-ILK/'
  '4wks': '4wks',
  # '16wks': '16wks\\',
  'ILK': 'ILK'
}

# model parameters
nums_epochs = 15
fit_steps = 500
channel_dims = 1
out_channels = 1
device = DEVICE[env]
batch_size = 2
start_filters = 64

# loss functions
loss_func = 'IoU'
cross_entropy_weight = 0.095

if loss_func == 'WCE':
  out_channels = 2
  weights = [1, cross_entropy_weight]
  class_weights = torch.FloatTensor(weights).cuda()
  criterion = torch.nn.CrossEntropyLoss(weight=class_weights) #CrossEntropyLoss
elif loss_func == 'IoU':
  criterion = IoULoss()
elif loss_func == 'Dice':
  criterion = DiceLoss()
elif loss_func == 'Focal':
  criterion = FocalLoss()
elif loss_func == 'Tversky':
  criterion = TverskyLoss()
else:
  print(colored('please use a valid loss function', 'red'))

# prediction parameters
# models = ['5']
models = ['15']
import torch.utils.data 
import torch
from tochvision import transforms
import numpy as np

class GainData(torch.utils.data.Dataset)
  def __init__(self, property, order, transform=None, target_transform=None, train=True):
    self.property = property
    self.train = train
    self.order = order
    self.transform = tranform
    self.target_transform = target_tranform
    #historydata
    if property==0:
      self.rawdata = np.load('history_data.npy')
      self.rawdata = self.rawdata.reshape(-1,-1,3,32,32)
      self.rawdata = self.rawdata.transpose((0,1,3,4,2))  
      self.rawlabel = np.load('history_label.npy')    
    #newdata
    if property==1:
      self.rawdata = np.load('history_data.npy')
      self.rawdata = self.rawdata.reshape(-1,-1,3,32,32)
      self.rawdata = self.rawdata.transpose((0,1,3,4,2)) 
      self.rawlabel = np.load('new_label.npy')
    #testdata
    if property==2:
      self.rawdata = []
      for i in range(9):
        self.rawdata.append(np.load('test_data{0}.npy'.format(i)).reshape(-1,3,32,32).transpose((0,2,3,1)))
      self.rawlabel = []
      for i in range(9):
        self.rawlabel.append(np.load('test_label{0}.npy'.format(i)))
        
  def __getitem__(self, idx):
    img = self.rawdata[order][idx]
    img = torch.from_numpy(img).float()
    label = self.rawlabel[order][idx]
     
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      label = self.target_transform(label)
    return img,label
    
  def __len__(self):
    return self.rawdata[order].shape[0]
     

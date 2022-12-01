from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import random
import os
from scipy.io import wavfile
import librosa
import scipy.io as scio

esp=1e-8

def norm(sig):
    x=sig-np.mean(sig)
    val=np.max(np.abs(x))
    if val>1e-8:
        x=x/val
    return x

def add_noise(sig,data='white',k=0.1):
    data_dir=f'data/noise/{data}.mat'
    mat=scio.loadmat(data_dir)
    #mat为dict
    data1=mat[data]
    data1.shape=(data1.shape[0],)
    noise=data1[0:sig.shape[0]]
    noise=np.array(noise)
    noise=noise.astype(np.float32)
    noise=norm(noise)
    # print(noise.shape)
    sig1=norm(sig+k*noise)
    return sig1

class SepformerDataset(Dataset):
    def __init__(self,data_path,type='train',sample_rate=8000,addnoise=False) -> None:
        super(SepformerDataset,self).__init__()
        file = ["mix", "human", "bgm"]
        self.mix_dir=os.path.join(data_path, file[0])
        self.mix_list = os.listdir(os.path.abspath(self.mix_dir))
        self.human_dir=os.path.join(data_path, file[1])
        self.human_list = os.listdir(os.path.abspath(self.human_dir))
        self.bgm_dir=os.path.join(data_path, file[2])
        self.bgm_list = os.listdir(os.path.abspath(self.bgm_dir))
        self.sample_rate=sample_rate
        self.addnoise=addnoise
        self.noiselist=os.listdir('data/noise')

    def __getitem__(self, index):
        mix_path=os.path.join(self.mix_dir,self.mix_list[index])
        sample_rate,mix_data=wavfile.read(mix_path)
        mix_data=norm(mix_data.astype(np.float32))
        if sample_rate!=self.sample_rate:
            mix_data=librosa.resample(mix_data,sample_rate,self.sample_rate)
        x=random.randint(0, 9)
        if x<7 and self.addnoise==True:
            y=random.randint(0, len(self.noiselist)-1)
            mix_data=add_noise(mix_data,self.noiselist[y].split('.')[0],random.uniform(0.05, 0.2))
        human_path=os.path.join(self.human_dir,self.mix_list[index])
        sample_rate,human_data=wavfile.read(human_path)
        human_data=norm(human_data.astype(np.float32))
        if sample_rate!=self.sample_rate:
            human_data=librosa.resample(human_data,sample_rate,self.sample_rate).astype(np.float32)
        
        bgm_path=os.path.join(self.bgm_dir,self.mix_list[index])
        sample_rate,bgm_data=wavfile.read(bgm_path)
        bgm_data=norm(bgm_data.astype(np.float32))
        if sample_rate!=self.sample_rate:
            bgm_data=librosa.resample(bgm_data,sample_rate,self.sample_rate).astype(np.float32)
        
        sep_data = np.stack((human_data, bgm_data), axis=0)

        return torch.from_numpy(mix_data),torch.from_numpy(sep_data),self.mix_list[index]
    
    def __len__(self):
        return len(self.mix_list)



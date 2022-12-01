import torch
from torch.utils.data import Dataset, DataLoader
from model.sepformer import Sepformer
import speechbrain
from scipy.io import wavfile
import os
import numpy as np
from tensorboardX import SummaryWriter
import librosa
esp=1e-4
def norm(sig):
    x=sig-np.mean(sig)
    val=np.max(np.abs(x))
    if val>1e-4:
        x=x/val
    return x
need_sample_rate=8000

path='data/下雨了薛之谦.wav'
savepath='data/testdata'
# sepdataset=dataset.SepformerDataset(path)
# seploader=DataLoader(dataset=sepdataset,batch_size=1,shuffle=False,num_workers=0,drop_last=False)
model = Sepformer(pertrain=True,pertrainpath='./checkpoint/finalMIR-1K27.pth')
# device='cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sample_rate, sig = wavfile.read(path)
# sig=norm(sig[300000:600000])
if sig.ndim>=2:
    sig=sig[:,0]
sig=norm(sig).astype(np.float32)
if sample_rate!=8000:
    sig=librosa.resample(y=sig,orig_sr=sample_rate,target_sr=need_sample_rate)
if sig.shape[0]>600000:
    sig=sig[490000:700000]
    sig=norm(sig)
print(sig.shape)
model.to(device)
mix=torch.from_numpy(sig)
mix=torch.unsqueeze(mix,dim=0).to(device)
print(sample_rate)
model.eval()
with torch.no_grad():
    sepdata=model(mix)
    # with SummaryWriter(comment='modelnet') as w:
    #     w.add_graph(model, (mix, ))
#     print(speechbrain.nnet.losses.get_si_snr_with_pitwrapper(soruce,sepdata))
    A,B=sepdata[0,:,0],sepdata[0,:,1]
    wavfile.write(os.path.join(savepath,'下雨了薛之谦A1.wav'),8000,A.cpu().numpy())
    wavfile.write(os.path.join(savepath,'下雨了薛之谦B1.wav'),8000,B.cpu().numpy())
#     wavfile.write(os.path.join(savepath,'mix.wav'),8000,mix.cpu().numpy())
# print(device)
# model.to(device)
# for i, (data) in enumerate(seploader):
#     mixdata, sourcedata ,mixture_lengths= data
#     mixdata, sourcedata,mixture_lengths=mixdata.to(device), sourcedata.to(device),mixture_lengths.to(device)
#     perdata=model(mixdata)
#     sourcedata=sourcedata.permute(0, 2, 1)
#     print(sourcedata.shape,perdata.shape)
#     loss=speechbrain.nnet.losses.get_si_snr_with_pitwrapper(sourcedata,perdata)
#     # # max_snr, estimate_source, reorder_estimate_source = cal_loss_pit(sourcedata,
#     # #                                                                                sourcedata,
#     # #                                                                                mixture_lengths)
#     print(loss)
#     # print(mixdata.shape)
#     # print(sourcedata.shape)
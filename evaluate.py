import argparse
import numpy as np
import speechbrain
import torch
from dataset.dataset import SepformerDataset
from model.sepformer import Sepformer
import json5
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def main(config):
    total_SISNRi = 0.
    total_cnt = 0.
    # 加载模型
    if config["model"] == "sepformer":
        model = Sepformer(pertrain=True,pertrainpath=config["model_path"])
    else:
        print("No loaded model!")
    model.eval()  # 将模型设置为验证模式
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 加载数据
    dataset = SepformerDataset(data_path=config["evaluate_dataset"]["data_dir"],
                              sample_rate=config["evaluate_dataset"]["sample_rate"],)

    data_loader=DataLoader(dataset=dataset,
                batch_size=config["evaluate_dataset"]["batch_size"],drop_last=False)
    # 不计算梯度
    with torch.no_grad():
        with tqdm(data_loader,unit='batch') as allbatch:
            for eachbatch in allbatch:
                allbatch.set_description(f"Test")
                mixdata, sourcedata,name= eachbatch
                mixdata = mixdata.to(device)
                sourcedata = sourcedata.permute(0, 2, 1).to(device)
                estimate_source = model(mixdata)  # 将数据放入模型
                loss=speechbrain.nnet.losses.get_si_snr_with_pitwrapper(sourcedata,estimate_source)
                total_cnt+=mixdata.shape[0]
                total_SISNRi-=torch.sum(loss)
                allbatch.set_postfix(SISNRi='{:.6f}'.format(-loss.mean()))
    print("Average SI_SNR improvement: {0:.2f}".format(total_SISNRi/total_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Speech Separation Performance")
    parser.add_argument("-C","--configuration",default="./config/test/evaluate.json5",type=str,help="Configuration (*.json).")
    args = parser.parse_args()
    configuration = json5.load(open(args.configuration))
    main(configuration)

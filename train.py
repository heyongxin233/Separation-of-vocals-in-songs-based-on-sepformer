import argparse
import torch
from dataset.dataset import SepformerDataset
from src.trainer import Trainer
from model.sepformer import Sepformer
import json5
import numpy as np
from adamp import AdamP, SGDP
from torch.utils.data import Dataset, DataLoader


def main(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 数据
    tr_dataset = SepformerDataset(data_path=config["train_dataset"]["train_dir"],
                              sample_rate=config["train_dataset"]["sample_rate"],addnoise=True)

    cv_dataset = SepformerDataset(data_path=config["validation_dataset"]["validation_dir"],
                              sample_rate=config["validation_dataset"]["sample_rate"],)

    tr_loader=DataLoader(dataset=tr_dataset,
                batch_size=config["train_loader"]["batch_size"],
                shuffle=config["train_loader"]["shuffle"],
                num_workers=config["train_loader"]["num_workers"],drop_last=False)

    cv_loader=DataLoader(dataset=cv_dataset,
                batch_size=config["validation_loader"]["batch_size"],
                shuffle=config["validation_loader"]["shuffle"],
                num_workers=config["validation_loader"]["num_workers"],drop_last=False)
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}
    # 模型
    if config["model"]["type"] == "sepformer":
        model = Sepformer(pertrain=config["model"]["pertrain"],
                          pertrainpath=config["model"]["ckptpath"],
                          N=config["model"]["sepformer"]["N"],
                          C=config["model"]["sepformer"]["C"],
                          L=config["model"]["sepformer"]["L"],
                          H=config["model"]["sepformer"]["H"],
                          K=config["model"]["sepformer"]["K"],
                          Global_B=config["model"]["sepformer"]["Global_B"],
                          Local_B=config["model"]["sepformer"]["Local_B"])
    else:
        print("No loaded model!")
        return
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.save(model,'./ckpt/basesepformer.pth')
    # print("save model over!!!!!!")
    # return 
    model.to(device)
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)
    #     model.cuda()

    if config["optimizer"]["type"] == "sgd":
        optimize = torch.optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"]["sgd"]["lr"],
            momentum=config["optimizer"]["sgd"]["momentum"],
            weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimize = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"], config["optimizer"]["adam"]["beta2"]))
    elif config["optimizer"]["type"] == "sgdp":
        optimize = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimize = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"], config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )
    else:
        print("Not support optimizer")
        return
    trainer=Trainer(data,model,optimize,config)
    for name, param in model.named_parameters():
        if param.requires_grad==False:
            param.requires_grad=True
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")
    parser.add_argument("-C","--configuration",default="./config/train/train.json5",type=str,
                        help="Configuration (*.json).")
    args = parser.parse_args()
    configuration = json5.load(open(args.configuration))
    main(configuration)

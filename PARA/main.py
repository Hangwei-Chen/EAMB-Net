import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
import datetime
from torch import nn
from models.EAMBNet import EAMBNet
from dataset import PARADataset
from util import AverageMeter
import option
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')

opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))

def adjust_learning_rate(params, optimizer, epoch):
    lr = params.init_lr * (0.5 ** ((epoch) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_PARA_save_csv, 'PARA-GiaaTrain.csv')
    test_csv_path = os.path.join(opt.path_to_PARA_save_csv, 'PARA-GiaaTest.csv')
    train_ds = PARADataset(train_csv_path, opt.path_to_PARA_images, if_train = True)
    test_ds = PARADataset(test_csv_path, opt.path_to_PARA_images, if_train=False)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.train_num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.test_num_workers, shuffle=False)
    return train_loader, test_loader


def train(opt,model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()
    for param in model.emotion_model.parameters():
        param.requires_grad = False
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        y = y.to(opt.device)
        y_pred = model(x)
        loss = criterion(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))
        if writer is not None:
            writer.add_scalar("train_loss", train_losses.avg, global_step=global_step + idx)
    return train_losses.avg


def validate(opt,model,  loader, criterion):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.to(opt.device)
            y = y.type(torch.FloatTensor)
            y = y.to(opt.device)
            y_pred = model(x)
            pscore_np = y_pred.data.cpu().numpy().astype('float')
            tscore_np = y.data.cpu().numpy().astype('float')
            pred_score += pscore_np.mean(axis=1).tolist()
            true_score += tscore_np.mean(axis=1).tolist()
            loss = criterion(y, y_pred)
            validate_losses.update(loss.item(), x.size(0))

        srcc_mean, _ = spearmanr(pred_score, true_score)
        lcc_mean, _ = pearsonr(pred_score, true_score)
        rmse = np.sqrt(((np.array(pred_score) - np.array(true_score)) ** 2).mean(axis=None))
        # mse = ((np.array(pred_score) - np.array(true_score)) ** 2).mean(axis=None)
        mae = (abs(np.array(pred_score) - np.array(true_score))).mean(axis=None)

        true_score = np.array(true_score)
        true_score_lable = np.where(true_score <= 0.6, 0, 1)  #scale[1,5]->[0.2,1]
        pred_score = np.array(pred_score)
        pred_score_lable = np.where(pred_score <= 0.6, 0, 1)
        acc = accuracy_score(true_score_lable, pred_score_lable)
        pred = (pred_score * 5).tolist()
        print('lcc_mean %4.4f,\tsrcc_mean %4.4f,\tacc %4.4f,\tMAE %4.4f,\tMSE %4.4f,\tRMSE %4.4f' % (
            lcc_mean, srcc_mean, acc, validate_losses.avg, mae, rmse))

    return validate_losses.avg, acc, lcc_mean, srcc_mean


def start_train(opt):
    train_loader,  test_loader = create_data_part(opt)
    model = EAMBNet()
    model = model.to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    criterion =  nn.MSELoss()
    criterion.to(opt.device)
    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer, criterion=criterion)
        test_loss, tacc, tlcc, tsrcc = validate(opt, model=model, loader=test_loader, criterion=criterion)

if __name__ =="__main__":

    #### train model
    start_train(opt)

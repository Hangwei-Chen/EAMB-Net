import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from models.EAMBNet import EAMBNet
from dataset import AVADataset
from util import EMD2Loss,EMD1Loss, AverageMeter
import option
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')

opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))


def adjust_learning_rate(params, optimizer, epoch):
    if epoch < 3:
        lr = params.init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = params.init_lr * (0.5 ** (epoch-2))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_score(opt,y_pred):
    w = torch.from_numpy(np.linspace(1,10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_AVA_save_csv, 'train.csv')
    val_csv_path = os.path.join(opt.path_to_AVA_save_csv, 'val.csv')
    test_csv_path = os.path.join(opt.path_to_AVA_save_csv, 'test.csv')
    train_ds = AVADataset(train_csv_path, opt.path_to_AVA_images, if_train = True)
    val_ds = AVADataset(val_csv_path, opt.path_to_AVA_images, if_train = False)
    test_ds = AVADataset(test_csv_path, opt.path_to_AVA_images, if_train=False)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.train_num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, num_workers=opt.test_num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.test_num_workers, shuffle=False)
    return train_loader, val_loader,test_loader


def train(opt,model, loader, optimizer, criterion):
    model.train()
    train_losses = AverageMeter()
    for param in model.emotion_model.parameters():
        param.requires_grad = False
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        y = y.to(opt.device)
        y_pred = model(x)
        loss = criterion(p_target=y, p_estimate=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))
        # if writer is not None:
        #     writer.add_scalar("train_loss", train_losses.avg, global_step=global_step + idx)
    return train_losses.avg


def validate(opt,model,  loader, criterion, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    EMD1_losses = AverageMeter()
    true_score = []
    pred_score = []
    criterion1 = EMD1Loss()
    criterion1.to(opt.device)
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.to(opt.device)
            y = y.type(torch.FloatTensor)
            y = y.to(opt.device)
            y_pred = model(x)
            pscore, pscore_np = get_score(opt,y_pred)
            tscore, tscore_np = get_score(opt,y)
            pred_score += pscore_np.tolist()
            true_score += tscore_np.tolist()
            loss = criterion(p_target=y, p_estimate=y_pred)
            EMD1= criterion1(p_target=y, p_estimate=y_pred)
            validate_losses.update(loss.item(), x.size(0))
            EMD1_losses.update(EMD1.item(), x.size(0))

            # if writer is not None:
            #     writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)
    srcc_mean, _ = spearmanr(pred_score, true_score)
    lcc_mean, _ = pearsonr(pred_score, true_score)
    rmse = np.sqrt(((np.array(pred_score) - np.array(true_score)) ** 2).mean(axis=None))
    mse = ((np.array(pred_score) - np.array(true_score)) ** 2).mean(axis=None)
    mae = (abs(np.array(pred_score) - np.array(true_score))).mean(axis=None)
    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 5.0, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5.0, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)

    print('PLCC %4.4f,\tSRCC %4.4f,\tAcc %4.4f,\tEMD1 %4.4f, \tEMD2 %4.4f,\tMSE %4.4f,\tMAE %4.4f,\tRMSE %4.4f' % (lcc_mean, srcc_mean, acc, EMD1_losses.avg,validate_losses.avg, mse,mae, rmse))

    return validate_losses.avg, acc, lcc_mean, srcc_mean



def start_train(opt):
    train_loader, val_loader, test_loader = create_data_part(opt)
    model = EAMBNet()
    model = model.to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    criterion = EMD2Loss()
    criterion.to(opt.device)
    # writer = SummaryWriter(log_dir=os.path.join(opt.experiment_dir_name, 'resluts'))

    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)

        train_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer, criterion=criterion)
        val_loss,vacc,vlcc,vsrcc = validate(opt,model=model, loader=val_loader, criterion=criterion)
        test_loss, tacc, tlcc, tsrcc = validate(opt, model=model, loader=test_loader, criterion=criterion)



if __name__ =="__main__":
    start_train(opt)

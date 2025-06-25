import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

from utils_harmonic import *
from msnet_stu import MSNet
from ftanet_tch import FTAnet

import time
import matplotlib.pyplot as plt


import argparse


class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def cgl_loss(pred_student, pred_teacher, target_mask, eps=1.5, temperature = 1.0):
    pred_student = F.softmax(pred_student / temperature, dim=1)
    pred_teacher = F.softmax(pred_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student + 1e-8), dim=-3)
    return loss.mean()

def train(train, test, epoch_num, batch_size, lr, gid, op, semi, pretrained_t=None, pretrained_s=None):
    print('batch size:', batch_size)
    torch.backends.cudnn.enabled = False

    gid = list(map(int, gid.split(",")))

    device = torch.device("cuda")


    Teacher = FTAnet()
    Net = MSNet()

    Teacher = torch.nn.DataParallel(Teacher, device_ids=gid)
    if pretrained_t is not None:
        Teacher.load_state_dict(torch.load(pretrained_t))

    Net = torch.nn.DataParallel(Net, device_ids=gid)
    if pretrained_s is not None:
        Net.load_state_dict(torch.load(pretrained_s))

    if gid is not None:
        Teacher.to(device=device)
        Net.to(device=device)
    else:
        Teacher.cpu()
        Net.cpu()
    Teacher.float()
    Net.float()


    epoch_num = epoch_num
    lr = lr

    X_train, y_train = load_train_data(path=train)
    test_list = load_list(path=test, mode='test')
    semi_list_all = load_list(path=semi, mode='test')
    semi_list = []
    y_train_harm = build_harmonic(f02img(y_train))
    with torch.no_grad():
        print('load_semi_list')
        for k in range(len(semi_list_all)):
            semi_list.append(load_onlyx_data(semi_list_all[k]))
        print('semi data loaded:', len(semi_list))





    best_epoch = 0
    best_OA = 0
    time_series = np.arange(64) * 0.01

    BCELoss = nn.BCEWithLogitsLoss()
    KLLoss = nn.KLDivLoss()
    CrossLoss = nn.CrossEntropyLoss()
    opt = optim.Adam(Net.parameters(), lr=lr)
    tick = time.time()

    Net.eval()
    for epoch in range(epoch_num):

        X_semi_list = []
        Y_semi_list = []
        with torch.no_grad():
            for k in range(len(semi_list)):
                X_semi = semi_list[k]
                predict = Net(X_semi.cuda())[0]
                predict = predict.detach().cpu()

                pred_index = torch.argmax(predict, dim=-2)
                pred_index[:, 1, :] -= 60
                pred_index[:, 2, :] += 60
                for token in range(X_semi.size(0)):
                    count = 0
                    for i in range(X_semi.size(-1)):
                        if pred_index[token, 0, i] == pred_index[token, 1, i] and pred_index[token, 0, i] == pred_index[
                            token, 2, i]:
                            count += 1
                    if count > X_semi.size(-1) * 0.75:
                        X_semi_list.append(X_semi[token, :, :, :].unsqueeze(dim=0))
                        Y_semi_list.append(predict[token, :, :, :].unsqueeze(dim=0))

        if len(X_semi_list) != 0:
            data_set = Dataset(data_tensor=torch.cat((X_train, torch.cat(X_semi_list, dim=0)), dim=0),
                               target_tensor=torch.cat(
                                   (y_train_harm, torch.cat(Y_semi_list, dim=0).cpu()), dim=0))
            data_loader = Data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, drop_last=True)

        else:
            data_set = Dataset(data_tensor=X_train, target_tensor=y_train_harm.cpu())
            data_loader = Data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, drop_last=True)

        print("dataset over")


        tick_e = time.time()
        Net.train()
        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):

            opt.zero_grad()
            if gid is not None:
                with torch.no_grad():
                    pred_t = Teacher(batch_x.to(device))
                pred_s = Net(batch_x.to(device))
                cgl_losses = 0
                for i in range(2, len(pred_s)):
                    cgl_losses += cgl_loss(pred_s[i], pred_t[0], batch_y.to(device))

                loss = CrossLoss(pred_s[0], batch_y.to(device).argmax(dim=1)) + cgl_loss(pred_s[0], pred_t[0],batch_y.to(device)) + 0.1 * cgl_losses + KLLoss(
                    torch.log(pred_s[0] + 1e-8), pred_t[0])

            else:
                with torch.no_grad():
                    pred_t = Teacher(batch_x.cpu())
                pred_s = Net(batch_x.cpu())
                cgl_losses = 0
                for i in range(2, len(pred_s)):
                    cgl_losses += cgl_loss(pred_s[i], pred_t[0], batch_y.cpu())

                loss = CrossLoss(pred_s[0], batch_y.cpu().argmax(dim=1)) + cgl_loss(pred_s[0], pred_t[0], batch_y.cpu()) + 0.1 * cgl_losses + KLLoss(
                    torch.log(pred_s[0] + 1e-8), pred_t[0])
            loss.backward()
            opt.step()
            train_loss += loss.item()


        Net.eval()
        eval_arr = np.zeros(5, dtype=np.double)
        with torch.no_grad():
            for i in range(len(test_list)):
                X_test, y_test = load_data(test_list[i])
                if gid is not None:
                    pred = Net(X_test.cuda())
                else:
                    pred= Net(X_test)
                est_freq = pred2res(pred[0]).flatten()
                ref_freq = y2res(y_test).flatten()
                time_series = np.arange(len(ref_freq)) * 0.01
                eval_arr += melody_eval(time_series, ref_freq, time_series, est_freq)

            eval_arr /= len(test_list)
            train_loss /= step + 1

        # scheduler.step()

        print("----------------------")
        print("Epoch={:3d}\tTrain_loss={:6.4f}\tLearning_rate={:6.4f}e-4".format(epoch, train_loss, 1e4 *
                                                                                 opt.state_dict()['param_groups'][0][
                                                                                     'lr']))
        print("Valid: VR={:.2f}\tVFA={:.2f}\tRPA={:.2f}\tRCA={:.2f}\tOA={:.2f}".format(eval_arr[0], eval_arr[1],
                                                                                       eval_arr[2], eval_arr[3],
                                                                                       eval_arr[4]))
        if eval_arr[-1] > best_OA:
            best_OA = eval_arr[-1]
            best_epoch = epoch

        torch.save(Net.state_dict(), op + '{:.2f}_{:d}'.format(eval_arr[4], epoch))
        print('Best Epoch: ', best_epoch, ' Best OA: ', best_OA)
        print("Time: {:5.2f}(Total: {:5.2f})".format(time.time() - tick_e, time.time() - tick))


def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-train', '--train_list_path',
                   help='the path of training data list (default: %(default)s)',
                   type=str, default='./training_set.txt')
    p.add_argument('-test', '--test_list_path',
                   help='the path of test data list (default: %(default)s)',
                   type=str, default='./test_04_npy.txt')
    p.add_argument('-semi', '--semi_list_path',
                   help='the path of semi data list (default: %(default)s)',
                   type=str, default='./semi_training_set.txt')
    p.add_argument('-ep', '--epoch_num',
                   help='the number of epoch (default: %(default)s)',
                   type=int, default=200)
    p.add_argument('-bs', '--batch_size',
                   help='The number of batch size (default: %(default)s)',
                   type=int, default=32)
    p.add_argument('-lr', '--learning_rate',
                   help='the number of learn rate (default: %(default)s)',
                   type=float, default=0.0008)
    p.add_argument('-gpu', '--gpu_index',
                   help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
                   type=str, default="0,1")
    p.add_argument('-o', '--output_dir',
                   help='Path to output folder (default: %(default)s)',
                   type=str, default='./model/')
    p.add_argument('-pm', '--pretrained_model',
                   help='the path of pretrained model (Transformer or Streamline) (default: %(default)s)',
                   type=str)

    return p.parse_args()


if __name__ == '__main__':
    args = parser()
    gid = args.gpu_index
    gid = list(map(int, gid.split(",")))[0]

    pretrained_t = None
    pretrained_s = None

    seed = 3643744328
    set_seed(seed)

    train(args.train_list_path, args.test_list_path, args.epoch_num, args.batch_size, args.learning_rate,
              args.gpu_index, args.output_dir, args.semi_list_path, pretrained_t=pretrained_t, pretrained_s = pretrained_s)

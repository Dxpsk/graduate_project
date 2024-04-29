import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
import tqdm
from models.RF import getRF
import const_rf as const
import wandb
import d2l.torch as d2l


EPOCH = 30
BATCH_SIZE = 200
LR = 0.0005
if_use_gpu = 1
num_classes = const.num_classes
criterion = nn.CrossEntropyLoss()

def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    return train_X, train_y


def adjust_learning_rate(optimizer, echo):
    lr = LR * (0.2 ** (echo / EPOCH))
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr



def load(feature_file):
    x, y = load_data(feature_file)
    cnn = getRF(num_classes)

    train_x = torch.unsqueeze(torch.from_numpy(x), dim=1).type(torch.FloatTensor)
    train_x = train_x.view(train_x.size(0), 1, 2, -1)
    train_y = torch.from_numpy(y).type(torch.LongTensor)

    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    return cnn, train_loader


def train(net, train_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    wandb.watch(net)
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.001)
    net.train() #将net设置为训练模式
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        train_pbar = tqdm.tqdm(train_iter, position=0, leave=True)
        metric = d2l.Accumulator(3)
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y =x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            #这里随着每个batch显示的是这个batch的平均损失

            #然后再每一个epoch结束后，我们需要得到训练集和测试集的损失和预测准确率
            with torch.no_grad():
                metric.add(loss *x.shape[0], d2l.accuracy(outputs, y), x.shape[0])  #累加每个batch的准确率
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]  #一个epoch的平均loss,和对训练集的预测准确率
        wandb.log({'train_loss':train_l ,'train_acc':train_acc}, step=epoch+1)
        
    torch.save(net.state_dict(), '/home/xjj/projects/graduate_project/RF_walkie-talkie/model_cw') # Save your best model
    print('Saving model with loss {:.3f}...'.format(loss))
    wandb.finish()
    return train_l, train_acc


if __name__ == '__main__':
    # TODO: change the data file path
    device = torch.device(f'cuda:{6}')
    wandb.init(project='walkie-talkie_cw',
               name='1st_run',
               config={'learning_rate': LR,
                       'batch_size': BATCH_SIZE,
                       'epoch': EPOCH,
                       'num_classes': num_classes})
    defense = 'walkie-talkie_cw'
    feature_file = '/data/Deep_fingerprint/processed_RF_data/' + defense + '-' +'train' + '.npy'
    method = defense
    net, train_iter = load(feature_file)
    net.to(device)
    train(net, train_iter, wandb.config.epoch, wandb.config.learning_rate, device)
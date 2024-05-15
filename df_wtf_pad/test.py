# %%
import torch
import pickle
import numpy as np
from torch import nn 
import numpy as np
import wandb
import tqdm
import d2l.torch as d2l
import math
from model import make_model

# %%
criterion = nn.CrossEntropyLoss()

# %%
x = torch.randn((32, 1, 5000))
print(make_model(95)(x).shape)

# %%
device = torch.device(f'cuda:{0}')
def LoadDataWTFPADCW():

    print ("Loading WTFPAD dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '/data/Deep_fingerprint/wtf_pad/close_world/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='latin1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # Load testing data
    with open(dataset_dir + 'X_test_WTFPAD.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_test_WTFPAD.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='latin1'))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# %%
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataWTFPADCW()

# %%
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# %%
X_train = X_train[:, np.newaxis, :]
X_valid = X_valid[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]
print ("Data dimensions:")
print ("X: Training data's shape : ", X_train.shape)
print ("y: Training data's shape : ", y_train.shape)
print ("X: Validation data's shape : ", X_valid.shape)
print ("y: Validation data's shape : ", y_valid.shape)
print ("X: Testing data's shape : ", X_test.shape)
print ("y: Testing data's shape : ", y_test.shape)

X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_valid = torch.LongTensor(y_valid)
y_test = torch.LongTensor(y_test)
print(type(X_train), type(y_train))
print(type(X_valid), type(y_valid))

def choose_model(model, optimizer, learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    if optimizer=='adam':
        return torch.optim.Adam(model.parameters(), learning_rate, betas, eps, weight_decay)
    elif optimizer=='adamax':
        return torch.optim.Adamax(model.parameters(), learning_rate, betas, eps, weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), learning_rate)
    
    


train_iter, valid_iter = d2l.load_array((X_train, y_train), batch_size=128), d2l.load_array((X_valid, y_valid), batch_size=128, is_train=False)
# %%
def train(net=make_model(95).to(device), train_iter=train_iter, valid_iter=valid_iter):
    wandb.init()
    config = wandb.config
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    wandb.watch(net)
    optimizer = choose_model(net, config.optimizer, config.lr, (config.betas1, config.betas2), config.eps, config.weight_decay)
    for epoch in range(config.num_epochs):
        net.train() #将net设置为训练模式
        train_pbar = tqdm.tqdm(train_iter, position=0, leave=True)
        metric = d2l.Accumulator(3)
        best_loss =math.inf

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y =x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{config.num_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
            #这里随着每个batch显示的是这个batch的平均损失

            #然后再每一个epoch结束后，我们需要得到训练集和测试集的损失和预测准确率
            with torch.no_grad():
                metric.add(loss *x.shape[0], d2l.accuracy(outputs, y), x.shape[0])  #累加每个batch的准确率
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]  #一个epoch的平均loss,和对训练集的预测准确率
        
        valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
        
        loss_record = []
        for x, y in valid_iter:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = net(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())
        valid_l = sum(loss_record)/len(loss_record)
        wandb.log({'train_loss':train_l ,'train_acc':train_acc ,'valid_loss':valid_l,'valid_acc':valid_acc}, step=epoch+1)

        if valid_l < best_loss:
            best_loss = valid_l
            torch.save(net.state_dict(), '/home/xjj/projects/graduate_project/df_wtf_pad/wtf_pad_cw_model') # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
    
    # #还要对x_test做一下推理
    test_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=128, shuffle=False)
    net = make_model(95)
    state_dict = torch.load('/home/xjj/projects/graduate_project/df_wtf_pad/wtf_pad_cw_model')
    net.load_state_dict(state_dict)
    loss_record = []
    net.to(device)
    for x, y in valid_iter:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = net(x)
            loss = criterion(pred, y)
        loss_record.append(loss.item())
    test_l = sum(loss_record)/(len(loss_record)*config.batch_size)
    test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
    wandb.log({'test_loss':test_l ,'test_acc':test_acc})
    print('test_acc: ', test_acc,'   ', 'test_loss: ', test_l)
    wandb.finish()
    return train_l, train_acc, valid_l, valid_acc

# %%

# wandb.init(project='DF_wtf_pad_CW',
#            name='1st_run',
#            config={'batch_size': 128, 'lr': 0.002, 'num_epochs': 30,
#                    'eps': 1e-08, 'weight_decay': 0,
#                    'betas':(0.9, 0.999)})
# model = make_model(95)
# model = model.to(device)
# train(model, train_iter, valid_iter, wandb.config['num_epochs'], 
#       wandb.config['lr'], device, wandb.config['betas'], wandb.config['eps'], wandb.config['weight_decay'])


# #还要对x_test做一下推理
# test_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=128, shuffle=False)
# net = make_model(95)
# state_dict = torch.load('/home/xjj/projects/graduate_project/df_wtf_pad/wtf_pad_cw_model')
# net.load_state_dict(state_dict)
# loss_record = []
# accuracy_record = []
# net.to(device)
# for x, y in valid_iter:
#     x, y = x.to(device), y.to(device)
#     with torch.no_grad():
#         pred = net(x)
#         loss = criterion(pred, y)
#     loss_record.append(loss.item())
# test_l = sum(loss_record)/len(loss_record)
# test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
# print('test_acc: ', test_acc,'   ', 'test_loss: ', test_l)

#test_acc:  0.9838947368421053     test_loss:  0.13146244606624047

sweep_config = {"method": "random",
                "name": "sweep",
                "parameters": {
                    "batch_size": {"distribution": "q_uniform",
                                   "max": 256,
                                   "min": 16,
                                   "q": 16},
                    "lr": {"distribution": "q_uniform",
                                   "max": 0.01,
                                   "min": 0.001,
                                   "q": 0.001},
                    "num_epochs": {"values": [10, 20, 30, 40, 50]},
                    "eps": {"values": [1e-8]},
                    "weight_decay": {"values": [0]},
                    "betas1": {"values": [0.9]},
                    "betas2": {"values": [0.99]},
                    "optimizer":{"values":["sgd", "adam", "adamax"]}
                }
                }

sweep_id = wandb.sweep(sweep_config, project='DF_wtf_pad_CW')
wandb.agent(sweep_id, train)


# import wandb
# sweep_configuration = {
# 'method': 'grid',
# 'name': 'sweep',
# 'metric': {
# 'goal': 'minimize',
# 'name': 'val_loss'
# },
# 'parameters': {
# 'batch_size': {'values': [16]},
# 'epoch': {'values': [5, 10]},
# 'train_acc': {'values': [0.68, 0.90]},
# }
# }
# def train(config=None):
#     wandb.init()
#     config=wandb.config
#     wandb.log({"batch_size": config.batch_size})
#     wandb.log({"epoch": config.epoch})
#     wandb.log({"train_acc": config.train_acc})
#     wandb.log({"val_loss": 0.10})
#     wandb.finish()



# sweep_id = wandb.sweep(sweep=sweep_configuration, project="test-sweep")
# wandb.agent(sweep_id, train)
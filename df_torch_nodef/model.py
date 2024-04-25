from torch import nn 

def make_model(classes):
    filter_num = ['None',32,64,128,256]
    kernel_size = ['None',9,9,9,9]
    conv_stride_size = ['None',1,1,1,1]
    pool_stride_size = ['None',4,4,4,4]
    pool_size = ['None',8,8,8,8]
    net = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding=4),
        nn.BatchNorm1d(num_features=32),
        nn.ELU(alpha=1.0),
        nn.Conv1d(in_channels=32, out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding=4),
        nn.BatchNorm1d(num_features=32),
        nn.ELU(alpha=1.0),
        nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=2),
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=32, out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding=4),
        nn.BatchNorm1d(num_features=64),
        nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding=4),
        nn.BatchNorm1d(num_features=64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=3),
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=64, out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding=4),
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(),
        nn.Conv1d(in_channels=128, out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding=4),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=4),
        nn.Dropout(p=0.1),
        nn.Conv1d(in_channels=128, out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding=4),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding=4),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=4),
        nn.Dropout(p=0.1),
        nn.Flatten(start_dim=-2),
        nn.Linear(256 * 20, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.7),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, classes)
    )
    return net
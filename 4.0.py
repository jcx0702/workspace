import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import pickle
import numpy as np
class Feeder_base(torch.utils.data.Dataset):
    """ Feeder base """

    def __init__(self, data_path, mode, snr=0):
        self.data_path = data_path
        self.mode = mode
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        # data: [SNR[MFCC,device,label]]
        pass

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data label
        data = self.data[index]
        label = self.label[index]
        
        if self.mode == 'single':
            return data, label
        elif self.mode == 'double':
            # augmentation
            data1 = self._augs(data)
            data2 = self._augs(data)
            return [data1, data2]

class Feeder_snr(Feeder_base):
    """ Feeder for snr inputs """
    def load_data(self):

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.data = []
        self.label = []

        for i,snr in enumerate(data):
            for j in snr:
                self.data.append(j[0])
                self.label.append(i)

class Feeder_device(Feeder_base):
    """ Feeder for device inputs """
    
    def load_data(self):

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)[self.snr]
        
        self.data = []
        self.label = []

        for i in data:
            self.data.append(i[0])
            self.label.append(i[1])

class Feeder_label(Feeder_base):
    """ Feeder for label inputs """

    def load_data(self):

        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)[self.snr]
        
        self.data = []
        self.label = []

        for i in data:
            self.data.append(i[0])
            self.label.append(i[2])
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
def load_data(self):
    self.trainloader = torch.utils.data.DataLoader(
        batch_size=self.arg.batch_size,
        shuffle=True,
        num_workers=self.arg.num_worker,
        pin_memory=True)
    if self.arg.phase == 'test':
        self.testloader = torch.utils.data.DataLoader(
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            pin_memory=True)
class CNN(nn.Module):
    def __init__(self, channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.pro = projection_MLP(32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.pro(x)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 将维度从 (batch_size, seq_len, embed_dim) 转换为 (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 对序列维度取平均，得到 (batch_size, embed_dim)
        x = self.fc(x)
        return x

class CNN_Transformer(nn.Module):
    def __init__(self, channels, embed_dim, num_heads, num_layers, num_classes):
        super(CNN_Transformer, self).__init__()
        self.cnn = CNN(channels)
        self.transformer = Transformer(embed_dim, num_heads, num_layers, num_classes)

    def forward(self, x):
        batch_size, seq_len, channels, signal_len = x.size()
        x = x.view(batch_size * seq_len, channels, signal_len)  # 将数据展平为 (batch_size * seq_len, channels, signal_len)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)  # 恢复为 (batch_size, seq_len, embed_dim)
        x = self.transformer(x)
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = CNN_Transformer(**args.model_args)  # 使用 CNN_Transformer 作为 encoder
        self.use_momentum = True
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_ema_updater = EMA(beta=args.moving_average_decay)
        self.predictor = prediction_MLP(args.projection_hidden_size, args.projection_size, args.projection_hidden_size)

        self.tt = args.tt
        self.ot = args.ot

        # create the queue
        self.K = args.K
        self.register_buffer("queue", torch.randn(args.projection_hidden_size, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.encoder)
        self.update_moving_average()
        set_requires_grad(self.target_encoder, False)
        return self.target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder)

    def forward(self, x1, x2):
        target_encoder = self.get_target_encoder() if self.use_momentum else self.encoder

        z1 = self.encoder(x1, return_projection=True)
        z2 = self.encoder(x2, return_projection=True)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        t1 = target_encoder(x1, return_projection=True)
        t2 = target_encoder(x2, return_projection=True)

        loss1 = loss_fn(p1, t2.detach_())
        loss2 = loss_fn(p2, t1.detach_())

        loss = loss1 + loss2
        return loss.mean()

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

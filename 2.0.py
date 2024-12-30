import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# 定义自定义数据集
class MIMIIDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, duration=5):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.file_paths[idx], sr=self.sr, duration=self.duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S = librosa.power_to_db(S, ref=np.max)
        S = torch.tensor(S).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx]).long()
        return S, label

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# 加载和预处理数据
def load_data(file_paths, labels, sr=16000, duration=5):
    data = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S = librosa.power_to_db(S, ref=np.max)
        data.append(S)
    data = np.array(data)
    data = data[..., np.newaxis]
    return data, labels

# 主函数
def main():
    # 加载数据
    file_paths = [...]  # 替换为MIMII数据集文件路径
    labels = [...]      # 替换为对应的标签
    data, labels = load_data(file_paths, labels)
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_dataset = MIMIIDataset(train_data, train_labels)
    val_dataset = MIMIIDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    cnn_model = CNNModel()
    transformer_model = TransformerModel(input_dim=128, hidden_dim=256, num_heads=8, num_layers=3, num_classes=2)
    model = nn.Sequential(cnn_model, transformer_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

# 定义自定义数据集
class MIMIIDataset(Dataset):
    def __init__(self, file_paths, labels, sr=16000, duration=5):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.file_paths[idx], sr=self.sr, duration=self.duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S = librosa.power_to_db(S, ref=np.max)
        S = torch.tensor(S).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx]).long()
        return S, label

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def load_data(file_paths, labels, sr=16000, duration=5):
    data = []
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S = librosa.power_to_db(S, ref=np.max)
        data.append(S)
    data = np.array(data)
    data = data[..., np.newaxis]
    return data, labels

# 主函数
def main():
    # 加载数据
    file_paths = [...]  
    labels = [...]      
    data, labels = load_data(file_paths, labels)
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_dataset = MIMIIDataset(train_data, train_labels)
    val_dataset = MIMIIDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    cnn_model = CNNModel()
    transformer_model = TransformerModel(input_dim=128, hidden_dim=256, num_heads=8, num_layers=3, num_classes=2)
    model = nn.Sequential(cnn_model, transformer_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'conv_autoencoder.pth')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()

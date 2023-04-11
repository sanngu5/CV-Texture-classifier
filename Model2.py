import enum
from typing_extensions import dataclass_transform
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary

# === 이미지 패치에서 특징 추출 ===
train_dir = "./texture_data/train"
test_dir = "./texture_data/test"
classes = ['brick', 'grass', 'ground']

X_train = []            # train 데이터 list
Y_train = []            # train 라벨 list

PATCH_SIZE = 32         # 이미지 패치 사이즈 (test데이터셋과 같은 크기로 조정)
np.random.seed(1234)
for idx, texture_name in enumerate(classes):   
    image_dir = os.path.join(train_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)     # 이미지를 100*100으로 축소

        for _ in range(10):
            h = np.random.randint(100-PATCH_SIZE)
            w = np.random.randint(100-PATCH_SIZE)

            image_p = image_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]

            X_train.append(image_p)
            Y_train.append(idx)

X_train = np.array(X_train)/128 -1      # 픽셀을 그대로 네트워크에 넣을 것이므로 스케일링함
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)    # (300, 32, 32, 3)
print('train label: ', Y_train.shape)   # (300)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        X_test.append(image)
        Y_test.append(idx)

X_test = np.array(X_test)/128 -1        # 픽셀을 그대로 네트워크에 넣을 것이므로 스케일링함
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)      # (150, 32, 32, 3)
print('test label: ', Y_test.shape)     # (150)

class Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        sample = (image, label)
        return sample

# === 신경망 모델 클래스 ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, output_dim)
        self.droptout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droptout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)

        return out


# === 모델 선언 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 10
learning_rate = 0.001
n_epoch = 100

Train_data = Dataset(images=X_train, labels=Y_train)
Test_data = Dataset(images=X_test, labels=Y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
Testloader = DataLoader(Test_data, batch_size=batch_size)

net = MLP(32*32*3, 1024, 128, 3)
net.to(device)
summary(net, (32, 32, 3), device='cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

# === 학습 ===
for epoch in range(n_epoch):
    train_loss = 0.0
    evaluation = []
    for i, data in enumerate(Trainloader, 0):
        features, labels = data
        labels = labels.long().to(device)
        features = features.to(device)
        optimizer.zero_grad()

        outputs = net(features.to(torch.float))

        _, predicted = torch.max(outputs.cpu().data, 1)

        evaluation.append((predicted==labels.cpu()).tolist())
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss = train_loss / (i+1)
    evaluation = [item for sublist in evaluation for item in sublist]
    train_acc = sum(evaluation) / len(evaluation)

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    if(epoch+1) % 1 == 0:
        test_loss = 0.0
        evaluation = []
        for i, data in enumerate(Testloader, 0):
            features, labels = data
            labels = labels.long().to(device)

            features = features.to(device)

            outputs = net(features.to(torch.float))
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted==labels.cpu()).tolist())
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        test_loss = test_loss / (i+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        test_acc = sum(evaluation) / len(evaluation)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if (epoch+1) % 20 == 0:
            print('[%d, %d]\tloss: %.4f\tAccuracy: %.4f\t\tval-loss: %.4f\tval-Accuracy: %.4f'%(epoch+1, n_epoch, train_loss, train_acc, test_loss, test_acc))

# === 학습/테스트 loss/정확도 시각화 ===
plt.plot(range(len(train_losses)), train_losses, label='train_loss')
plt.plot(range(len(test_losses)), test_losses, label='test_loss')
plt.legend()
plt.show()

plt.plot(range(len(train_accs)), train_accs, label='train acc')
plt.plot(range(len(test_accs)), test_accs, label='test acc')
plt.legend()
plt.show()
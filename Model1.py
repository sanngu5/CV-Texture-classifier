from random import shuffle
from skimage.feature import graycomatrix, graycoprops
from scipy import signal as sg

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# === laws texture 계산 함수 ===
def laws_texture(gray_image):
    (rows, cols) = gray_image.shape[:2]
    smooth_kernel = (1/25)*np.ones((5,5))
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")
    gray_processed = np.abs(gray_image - gray_smooth)

    # 1D 필터
    filter_vectors = np.array([[1, 4, 6, 4, 1],     # L5
                            [-1, 2, 0, 2, 1],       # E5
                            [-1, 0, 2, 0, 1],       # S5
                            [1, -4, 6, -4, 1]])     # R5

    # 0: L5L5, 1: L5E5, 2: L5S5, 3: L5R5
    # 4: E5L5, 5: E5E5, 6: E5S5, 7: E5R5
    # 8: S5L5, 9: S5E5, 10:S5S5, 11:S5R5
    # 12:R5L5, 13:R5E5, 14:R5S5, 15:R5R5           

    
    # 행렬 곱으로 2D필터 생성
    filters = list()
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5, 1), 
                                    filter_vectors[j][:].reshape(1, 5)))

    # 계산된 convolution 결과 저장할 배열
    conv_maps = np.zeros((rows,cols,16))
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')

    # 9+1개 중요한 texture map 계산
    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2)     # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2)     # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2)    # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7]+conv_maps[:, :, 14])//2)    # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6]+conv_maps[:, :, 9])//2)     # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2)   # S5R5 / R5S5
    texture_maps.append((conv_maps[:, :, 10]))  # S5S5
    texture_maps.append((conv_maps[:, :, 5]))   # E5E5
    texture_maps.append((conv_maps[:, :, 15]))  # R5R5
    texture_maps.append((conv_maps[:, :, 0]))   # L5L5 (use to norm TEM)

    # Law's texture energy 계산
    TEM = list()
    for j in range(9):
        TEM.append(np.abs(texture_maps[j]).sum() / np.abs(texture_maps[9]).sum())

    return TEM

# === 이미지 패치에서 특징 추출 ===
train_dir = "./texture_data/train"
test_dir = "./texture_data/test"
classes = ['brick', 'grass', 'ground']

X_train = []            # train 데이터 list
Y_train = []            # train 라벨 list

PATCH_SIZE = 30         # 이미지 패치 사이즈
np.random.seed(1234)
for idx, texture_name in enumerate(classes):   
    image_dir = os.path.join(train_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)     # 이미지를 100*100으로 축소

        for _ in range(10):
            h = np.random.randint(100-PATCH_SIZE)
            w = np.random.randint(100-PATCH_SIZE)

            image_p = image_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]           # 영역 내에서 랜덤하게 패치를 추출
            image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)    # 이미지를 흑백으로 변환
            # X_train = GLCM 계산해서 구한 Numeric feature + Law's energy
            glcm = graycomatrix(image_p_gray, distances=[1], angles=[0], levels=256, 
                                symmetric=False, normed=True)
            X_train.append([graycoprops(glcm, 'dissimilarity')[0, 0], 
                            graycoprops(glcm, 'correlation')[0, 0]]
                            + laws_texture(image_p_gray))
            Y_train.append(idx)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print('train data: ', X_train.shape)    # (300, 11)
print('train label: ', Y_train.shape)   # (300)

X_test = []
Y_test = []

for idx, texture_name in enumerate(classes):   
    image_dir = os.path.join(test_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # 이미지를 흑백으로 변환
        # X_test = GLCM 계산해서 구한 Numeric feature(2차원) + Law's energy(9차원)
        glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256,
                            symmetric=False, normed=True)
        X_test.append([graycoprops(glcm, 'dissimilarity')[0, 0], 
                        graycoprops(glcm, 'correlation')[0, 0]]
                        + laws_texture(image_gray))
        Y_test.append(idx)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data: ', X_test.shape)    # (150, 11)
print('test label: ', Y_test.shape)   # (150)


# === 신경망에 필요한 모듈 ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

# === 데이터셋 클래스 ===
class textureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features    # 텍스쳐 특징
        self.labels = labels        # 텍스쳐 라벨

    def __len__(self):
        return len(self.labels)     # 데이터셋 크기 반환

    def __getitem__(self, idx):     # idx번째 샘플(특징, 라벨) 반환
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = self.features[idx]
        label = self.labels[idx]
        sample = (feature, label)

        return sample

# === 신경망 모델 클래스 ===
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim ,output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 10
learning_rate = 0.01
n_epoch = 500

Train_data = textureDataset(features=X_train, labels=Y_train)
Test_data = textureDataset(features=X_test, labels=Y_test)

Trainloader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
Testloader = DataLoader(Test_data, batch_size=batch_size)

net = MLP(11, 8, 3)
net.to(device)
summary(net, (11,), device='cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

# === 학습 ===
for epoch in range(n_epoch):
    train_loss = 0.0
    evaluation = []     # 예측 정확 여부 저장할 list
    net.train()
    for i, data in enumerate(Trainloader, 0):
        features, labels = data
        labels = labels.long().to(device)
        optimizer.zero_grad()

        outputs = net(features.to(torch.float))

        _, predicted = torch.max(outputs.cpu().data, 1)         # 출력의 제일 큰 값의 index 반환
        evaluation.append((predicted==labels.cpu()).tolist())   # 정답이면 True 추가 아니면 False 추가
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss/(i+1)
    # [True, False] 값을 list로 저장, 300차원, 리스트안 리스트가 있으므로 제거하여 True, False만 남게함
    evaluation = [item for sublist in evaluation for item in sublist] 
    train_acc = sum(evaluation)/len(evaluation)

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # === 테스트 ===
    if(epoch+1) % 1 == 0:
        test_loss = 0.0
        evaluation =[]
        net.eval()
        for i, data in enumerate(Testloader, 0):
            features, labels = data
            labels = labels.long().to(device)
            features = features.to(device)

            outputs = net(features.to(torch.float))
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted==labels.cpu()).tolist()) 

            loss = criterion(outputs, labels)
            test_loss += loss.item()
        test_loss = test_loss/(i+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        test_acc = sum(evaluation)/len(evaluation)

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
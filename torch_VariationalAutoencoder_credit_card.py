import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Based on:
# http://sofasofa.io/tutorials/anomaly_detection/



data = pd.read_csv('creditcard.csv')
num_nonfraud = np.sum(data['Class'] == 0)
num_fraud = np.sum(data['Class'] == 1)
# plt.bar(['Fraud', 'non-fraud'], [num_fraud, num_nonfraud], color='dodgerblue')
# plt.show()
print("number of fraud:", num_fraud)
print("number of non fraud:", num_nonfraud)

data = data.drop(['Time'], axis=1)
# 去掉Time, 对Amount进行标准化
data['Amount'] = StandardScaler().fit_transform(data[['Amount']])

print(data.shape)


# 提取负样本，并且按照8:2切成训练集和测试集
# class == 0: non fraud,
# class == 1: fraud

mask = (data['Class'] == 0)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=920)


X_train = data_train.drop(['Class'], axis=1).values
X_test = data_test.drop(['Class'], axis=1).values

y_train = data_train['Class'].values
y_test = data_test['Class'].values

# 提取所有正样本(被诈骗），作为测试集的一部分
data_fraud = data[~mask]
# 注意利用mask的技巧
X_fraud = data_fraud.drop(['Class'], axis=1).values
y_fraud = data_fraud['Class'].values

# print(X_fraud)
# print(y_fraud)

print("X_train size:", X_train.shape)
print("X_test size:", X_test.shape)
print("X_fraud size:", X_fraud.shape)


X_test = np.concatenate((X_test, X_fraud),0)
y_test = np.concatenate((y_test, y_fraud),0)


X_train=torch.from_numpy(np.array(X_train)).type(torch.FloatTensor)
# y_train=torch.from_numpy(np.array(y_train)).type(torch.FloatTensor)
X_fraud=torch.from_numpy(np.array(X_fraud)).type(torch.FloatTensor)
X_test=torch.from_numpy(np.array(X_test)).type(torch.FloatTensor)
y_test=torch.from_numpy(np.array(y_test)).type(torch.FloatTensor)


print("tensor X_train size:", X_train.shape)
print("tensor X_test size:", X_test.shape)
print("tensor X_fraud size:", X_fraud.shape)

print("tensor y_test size:", y_test.shape)

# print(X_train)

# train_Dataset = Data.TensorDataset(X_train,y_train)
test_Dataset = Data.TensorDataset(X_test,y_test)


train_loader = Data.DataLoader(
    dataset=X_train,
    batch_size=32,
    shuffle=True)

test_loader = Data.DataLoader(
    dataset=test_Dataset,
    batch_size=32,
    shuffle=True)

# 设置Autoencoder的参数
# 隐藏层节点数分别为16，8，8，16
# epoch为50，batch size为32
# num_epoch = 50
# batch_size = 32

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enconv =  nn.Sequential(
            nn.Conv1d(1, 16, 4, 1),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, 32 ,4, 1),
            nn.BatchNorm1d(32),
            nn.Tanh(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(736, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16,8),

        )

        self.decoder = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 736),
            nn.Tanh(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, 1),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.ConvTranspose1d(16, 1, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        en = self.enconv(x)
        encoded = self.encoder(en.view(en.size(0), -1))
        decoded = self.decoder(encoded)
        de = self.deconv(decoded.view(decoded.size(0), 32, 23))
        return de


autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
loss_func = nn.MSELoss()

for epoch in range(40):
    for step, x in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        x = x.view(-1, 1, 29)
        decoded = autoencoder(x)
        loss = loss_func(decoded, x)

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 1000 == 0:
            print("Epoch ", epoch, ' | train loss: %.4f ' % loss.data.numpy())

# 模型训练完成，可以进行测试
pred_test = autoencoder(X_test).data.numpy()
pred_fraud = autoencoder(X_fraud).data.numpy()

X_test = X_test.data.numpy()
X_fraud = X_fraud.data.numpy()

# 计算还原误差MSE和MAE
mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)


mse_df = pd.DataFrame()
mse_df['Class'] = [0] * len(mse_test) + [1] * len(mse_fraud)
mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
mse_df = mse_df.sample(frac=1).reset_index(drop=True)

# 分别画出测试集中正样本和负样本的还原误差MAE和MSE
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(14, 5))
plt.subplot(121)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MAE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.title('Reconstruction MAE')
plt.ylabel('Reconstruction MAE'); plt.xlabel('Index')
plt.subplot(122)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0], fontsize=12)
plt.title('Reconstruction MSE')
plt.ylabel('Reconstruction MSE')
plt.xlabel('Index')
plt.show()

# 画出Precision-Recall曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
    pr_auc = auc(recall, precision)
    plt.title('Precision-Recall curve based on %s\nAUC = %0.2f'%(metric, pr_auc))
    plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
    plt.xlabel('Recall'); plt.ylabel('Precision')
plt.show()

# 画出ROC曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i+1)
    fpr, tpr, _ = roc_curve(mse_df['Class'], mse_df[metric])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f'%(metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0,1],[0,1], c='dodgerblue', ls='--')
    plt.ylabel('TPR'); plt.xlabel('FPR')
plt.show()

# 画出MSE、MAE散点图
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(10, 5))
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp['MAE'],
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0])
plt.ylabel('Reconstruction RMSE'); plt.xlabel('Reconstruction MAE')
plt.show()

torch.save(autoencoder, "PytorchModel_autoencoder.pkl")





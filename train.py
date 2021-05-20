import time

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data

from model import SNet

start_time = time.time()

model = SNet()
Epoch = 5
batch_size = 64
lr = 0.001
train_data = torchvision.datasets.MNIST(root='./.data/', train=True, \
                                        transform=torchvision.transforms.ToTensor(), download=False)

train_loader = Data.DataLoader(train_data, batch_size=batch_size,\
                               shuffle=True, num_workers=0, drop_last=True)
#dataset为数据集，也可以用trochvision.datasets
#shuffle为是否打乱数据集，默认为True
#num_workers为使用的线程数，pin_memory为是否使用锁页内存，设置为True的话会训练的更快一些，默认为FALSE
#drop_last为是否丢弃最后不足batch_size的数据，默认为FALSE

loss_function = nn.CrossEntropyLoss()#使用交叉熵作为损失函数，因为是单标签分类
optmizer = torch.optim.Adam(model.parameters(), lr=lr)#优化器使用Adam
torch.set_grad_enabled(True)#开启自动求导
model.train()#启用Batch Normalization层和Dropout层
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)#设置cuda加速

for epoch in range(Epoch):
    running_loss = 0.0
    acc = 0.0
    for step, data in enumerate(train_loader):
        x,y = data
        optmizer.zero_grad()
        y_pred = model(x.to(device, torch.float))
        loss = loss_function(y_pred, y.to(device, torch.long))
        loss.backward()
        running_loss += float(loss.data.cpu())
        pred = y_pred.argmax(dim=1)
        acc += (pred.data.cpu() == y.data).sum()
        optmizer.step()
        if step % 100 == 99:
            loss_avg = running_loss / (step + 1)
            acc_avg = float(acc / ((step + 1) * batch_size))
            print('Epoch', epoch + 1, ',step', step + 1,\
                  '|Loss_avg: %.4f' % loss_avg, '|Acc_avg: %.4f' % acc_avg)

torch.save(model, './SNet.pkl')

end_time = time.time()
print("Time: ",end_time - start_time)


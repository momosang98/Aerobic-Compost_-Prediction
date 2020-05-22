import xlrd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import time

# Hyper Parameters
TIME_STEP = 3  # MLP
INPUT_SIZE = 16  # MLP
LR = 0.001  # learning rate

# 调取测试集
d = pd.read_excel(r'D:/python/program/testset.xlsx')
data = pd.DataFrame(d)

Day = np.array(data.iloc[0:16, 0], float)
mean = np.mean(Day)
std = np.std(Day)
D = (Day - mean) / std
D = D.reshape(1, 16)
D = np.array(D)

T = np.array(data.iloc[0:16, 1], float)
mean = np.mean(T)
std = np.std(T)
T = (T - mean) / std
T = T.reshape(1, 16)
T = np.array(T)

pH = np.array(data.iloc[0:16, 2], float)
mean = np.mean(pH)
std = np.std(pH)
pH = (pH - mean) / std
pH = pH.reshape(1, 16)
pH = np.array(pH)

GI = np.array(data.iloc[0:16, 3], float)
mean = np.mean(GI)
std = np.std(GI)
GIt = (GI - mean) / std
GIt = GIt.reshape(1, 16)

test_x = np.append(D, T, axis=1)
test_x = np.append(test_x, pH, axis=1)
test_x = test_x.reshape(3, 16)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(GIt, dtype=torch.float32)
# b_test_y = test_y.view(1, 1, 16)


np.set_printoptions(suppress=True, threshold=99999)


# 建立数据库
class MyDataset(Dataset):
    def __init__(self, excel_path):
        self.n_data, self.all_data = self.get_alldata(excel_path)

    def get_alldata(self, excel_path):
        # 打开文件，获取excel文件的workbook（工作簿）对象
        excel = xlrd.open_workbook(excel_path, encoding_override="utf-8")
        # 获取sheet对象,此处仅处理第一张sheet
        sheet = excel.sheets()[0]

        n_data = int((sheet.nrows - 1) / 16)

        all_data = []
        for index in range(sheet.nrows):
            sheet_cell_row = sheet.row_values(index)  # 获取指定行对象
            # print(sheet_cell_row)
            if index != 0:
                all_data.append(sheet_cell_row)
        all_data = np.array(all_data)

        return n_data, all_data

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        # 数据编号从0开始
        start = int(index * 16)
        end = int(start + 16)

        D = self.all_data[start:end, 0]
        mean = np.mean(D)
        std = np.std(D)
        D = (D - mean) / std
        D = D.reshape(1, 16)
        T = self.all_data[start:end, 1]
        mean = np.mean(T)
        std = np.std(T)
        T = (T - mean) / std
        T = T.reshape(1, 16)
        H = self.all_data[start:end, 2]
        mean = np.mean(H)
        std = np.std(H)
        H = (H - mean) / std
        H = H.reshape(1, 16)
        x = np.append(D, T, axis=1)
        x = np.append(x, H, axis=1)
        x = x.reshape(3, 16)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.all_data[start:end, 3]
        mean = np.mean(y)
        std = np.std(y)
        y = (y - mean) / std
        y = y.reshape(1, 16)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# 搭建模型
class MLP(nn.Module):
    def __init__(self, n_input, n_output):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, n_output)

    def forward(self, din):
        din = din.view(-1, 3 * 16)
        dout = torch.relu(self.fc1(din))
        dout = torch.relu(self.fc2(dout))
        dout = torch.relu(self.fc3(dout))
        dout = torch.tanh(self.fc4(dout))

        return dout

mlp = MLP(48, 16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

# 计时开始
time_start = time.time()
if __name__ == '__main__':
    excel_path = 'dataset.xlsx'

    my_dataset = MyDataset(excel_path)
    # print(my_dataset.n_data) # 11
    # print(my_dataset.all_data.shape) # (176,4)
    # x, y =my_dataset.__getitem__(10)
    # print(x.shape)
    # print(y.shape)

    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=1)
    a = []
    b = []
    c = []
    plt.ion()
    plt.show()
    for i in range(200):
        for i_batch, data_batch in enumerate(dataloader):
            '''
            data_batch包括了x和y
            x = data_batch[0], shape:(3,16)
            y = data_batch[1], shape:(1,16)
            '''
            # print(i_batch)
            x = data_batch[0]
            x = x.view(3, 16)#MLP
            y = data_batch[1]
            y = y.view(1, 16)#MLP
            output = mlp(x)
            loss = loss_func(output, y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        if (i + 1) % 1 == 0:
            tout = mlp(test_x)
            t_loss = loss_func(tout, test_y)
            print('Epoch:{}, train_loss:{:.5f},test_loss:{:.5f}'.format(i + 1, loss.item(), t_loss.item()))
            pout = output.view(-1).data.numpy()
            py = y.view(-1).data.numpy()
            plt.cla()
            '''a.append(loss.data.numpy())  # a是List
            l = np.array(a)  # list转numpy
            c.append(t_loss.data.numpy())
            C = np.array(c)
            b.append(i + 1)
            B = np.array(b)
            plt.subplot(121)
            plt.title("loss")
            plt.xlabel("epoch")
            plt.ylabel("regression loss")
            plt.plot(B, l, 'b-', label='train_loss')
            plt.plot(B, C, 'r-', label='test_loss')

            plt.legend(loc='best')'''


            plt.xlabel("num")
            plt.ylabel("GI")
            plt.plot(pout, c='r', linestyle='--', marker='o', label='prediction')
            plt.plot(py, c='b', linestyle='-.', marker='v', label='real')

            plt.title("prediction result(train_epoch200)")
            plt.legend(loc='best')
            errort = pout - py
            error_100 = errort / py
            print(error_100)


            plt.pause(0.1)

    plt.ioff()  # 结束实时画图
    time_end = time.time()
    print('totally cost', time_end - time_start)

    plt.show()

    pred = model(test_x)
    pred_test = pred.view(-1).data.numpy()
    train_size = int(len(pred_test))
    pred_train = pred_test[:train_size]
    # print(pred_train)

    plt.plot(pred_train, c='r', linestyle='--', marker='o', label='prediction')

    dataY = test_y.view(-1).data.numpy()
    plt.plot(dataY, c='b', linestyle='-.', marker='v', label='real')
    plt.xlabel("num")
    plt.ylabel("GI")
    plt.title("prediction result(test_epoch200)")
    plt.legend(loc='best')

    plt.show()
    error = pred_train - dataY
    error_100 = error / dataY
    print(error_100)


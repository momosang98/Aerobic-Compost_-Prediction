import xlrd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import time


np.set_printoptions(suppress=True, threshold=99999)
#调取测试集
d=pd.read_excel(r'D:/python/program/testset.xlsx')
data=pd.DataFrame(d)

Day=np.array(data.iloc[0:16,0],float)
mean=np.mean(Day)
std=np.std(Day)
D=(Day-mean)/std
D=D.reshape(16,1)
D=np.array(D)

T=np.array(data.iloc[0:16,1],float)
mean=np.mean(T)
std=np.std(T)
T=(T-mean)/std
T=T.reshape(16,1)
T=np.array(T)


pH=np.array(data.iloc[0:16,2],float)
mean=np.mean(pH)
std=np.std(pH)
pH=(pH-mean)/std
pH=pH.reshape(16,1)
pH=np.array(pH)


GI = np.array(data.iloc[0:16,3],float)
mean = np.mean(GI)
std = np.std(GI)
GIt = (GI-mean)/std
GIt = GIt.reshape(16,1)
GI = torch.tensor(GI, dtype=torch.float32)

x=np.append(D,T,axis=1)
x=np.append(x,pH,axis=1)
x=x.reshape(16,3)
test_x = torch.tensor(x, dtype=torch.float32)
test_y = torch.tensor(GIt, dtype=torch.float32)
b_test_x = test_x.view(1, 16, 3)
b_test_y = test_y.view(1, 16, 1)

#建立数据库
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
        D = D.reshape(16, 1)
        T = self.all_data[start:end, 1]
        mean = np.mean(T)
        std = np.std(T)
        T = (T - mean) / std
        T = T.reshape(16, 1)
        H = self.all_data[start:end, 2]
        mean = np.mean(H)
        std = np.std(H)
        H = (H - mean) / std
        H = H.reshape(16, 1)
        x = np.append(D, T, axis=1)
        x = np.append(x, H, axis=1)
        x = x.reshape(16, 3)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.all_data[start:end, 3]
        mean = np.mean(y)
        std = np.std(y)
        y = (y - mean) / std
        y = y.reshape(16, 1)
        y = torch.tensor(y, dtype=torch.float32)
        return x,y

#建立模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(  #
            input_size=3,
            hidden_size=128,     # rnn hidden unit
            num_layers=2,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, (h_n, h_c)  = self.lstm(x)   # h_state 也要作为 RNN 的一个输入

        outs = []    # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)

lstm = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)   # optimize all rnn parameters
loss_func = nn.MSELoss()

#计时开始
time_start=time.time()

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
            x = data_batch[0]
            y = data_batch[1]
            b_x = x.view(1, 16, 3)#LSTM
            b_y = y.view(1, 16, 1)#LSTM
            output = lstm(b_x)
            loss = loss_func(output, b_y)  # MSE loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        if (i + 1) % 1 == 0:

            tout = rnn(b_test_x)
            t_loss = loss_func(tout, b_test_y)
            print('Epoch:{}, train_loss:{:.5f},test_loss:{:.5f}'.format(i + 1, loss.item(),t_loss.item()))
            #pout = output.view(-1).data.numpy()
            #py = y.view(-1).data.numpy()
            plt.cla()
            a.append(loss.data.numpy())  # a是List
            l = np.array(a)  # list转numpy
            c.append(t_loss.data.numpy())
            C = np.array(c)
            b.append(i + 1)
            B = np.array(b)
            #plt.cla()
            plt.plot(B, l, 'b-', label='train_loss')
            plt.plot(B, C, 'r-', label='test_loss')
            plt.title("loss")
            plt.xlabel("epoch")
            plt.ylabel("regression loss")
            plt.legend(loc='best')
            plt.pause(0.1)
            '''plt.xlabel("num")
            plt.ylabel("GI")
            plt.plot(pout, c='r', linestyle='--', marker='o', label='prediction')
            plt.plot(py, c='b', linestyle='-.', marker='v', label='real')

            plt.title("prediction result (train epoch200)")
            plt.legend(loc='best')
            plt.pause(0.1)'''

    plt.ioff()  # 结束实时画图
    time_end = time.time()
    print('epoch:200,totally cost', time_end - time_start)
    plt.show()


    pred = rnn(b_test_x)
    pred_test = pred.view(-1).data.numpy()
    train_size = int(len(pred_test))
    pred_train = pred_test[:train_size]

    print(pred_train)
    plt.plot(pred_train, c='r', linestyle='--',marker='o',label='prediction')

    dataY = test_y.view(-1).data.numpy()
    plt.plot(dataY, c='b', linestyle='-.',marker='v', label='real')
    plt.xlabel("num")
    plt.ylabel("GI")
    plt.title("prediction result(test_epoch200)")
    plt.legend(loc='best')
    plt.show()
    error= pred_train - dataY
    error_100 = error/dataY
    print(error_100)

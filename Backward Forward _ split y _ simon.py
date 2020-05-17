import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/NanumGothic.ttf').get_name()
rc('font', family=font_name)
rc('axes', unicode_minus=False)

# this is one way to define a network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, layers):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden) for l in range(layers)])
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.input(x))      # activation function for hidden layer
        for layer in self.layers:
            x = F.relu(layer(x))
        result = self.predict(x)             # linear output
        return result

class Net_(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, layers):
        super(Net_, self).__init__()
        self.input = torch.nn.Linear(n_feature, n_hidden)   # 첫 레이어는 feature의 크기로
        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden) for l in range(layers)])
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.input(x))      # activation function for hidden layer
        for layer in self.layers:
            x = F.relu(layer(x))
        result = F.sigmoid(self.predict(x))             # linear output
        return result


class BackwardForward(nn.Module):
    def __init__(self, backward_model, forward_model):
        super(BackwardForward, self).__init__()
        self.backward_model = backward_model
        self.forward_model = forward_model
#         self.output = torch.nn.Linear(n_hidden,1)

    def forward(self, y_data):
        x1 = self.backward_model(y_data)
        x2 = self.forward_model(x1)
#         output = self.output(F.linear(x2))
        return x2

def generate_data(n_samples, elements):
    epsilon = np.random.normal(size=(n_samples))
    x_data = np.float32(np.random.uniform(-10.5, 10.5, n_samples))
#     y_data = np.sin(0.75 * x_data) * 7.0 + x_data * 0.5
    y_data = 7*np.sin(elements*x_data) + 1.5*x_data + epsilon*0.5
    formula = 'y = 7 sin({} x) + 1.5 x + 0.5 epsilon'.format(elements)

#     # 최원우 연구원 데이터
#     x_data = np.float32(np.random.uniform(-10.5, 10.5, n_samples))
#     y_data = np.sin(0.75 * x_data) * 7.0 + x_data * 0.5

    return x_data, y_data, formula

def train(net, X_train, y_train, X_valid, y_valid, optimizer, epoch) :

    loss_fn = nn.MSELoss()

    train_loss = []
    valid_loss = []

    for t in range(epoch) :

        optimizer.zero_grad()

        # train
        pred = net(X_train)
        loss = loss_fn(pred, y_train)
        train_loss.append(loss.item())

        # evaluate
        pred_eval = net(X_valid)
        loss_eval = loss_fn(pred_eval, y_valid)
        valid_loss.append(loss_eval.item())


        loss.backward()
        optimizer.step()

        if t % 500 == 0:
            print(t, 'train loss >>> ', loss.item(), 'valid loss >>> ', loss_eval.item())


    return train_loss, valid_loss

def get_data(X, y, scaler, n_input=1, n_output=1) :
    X_reshaped = np.float32(X).reshape(n_samples, n_input)
    y_reshaped = np.float32(y).reshape(n_samples, n_output)


    if scaler == 'minmaxScaler' :
        minmaxScaler = MinMaxScaler()
        minmaxScaler.fit(X_reshaped)
        X_reshaped = minmaxScaler.transform(X_reshaped)

    elif scaler == 'standardScaler' :
        standardScaler = StandardScaler()
        standardScaler.fit(X_reshaped)
        X_reshaped = standardScaler.transform(X_reshaped)


    X_train, X_tmp, y_train, y_tmp = train_test_split(X_reshaped, y_reshaped, test_size=0.3, random_state=1)
    X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=1)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_valid = torch.from_numpy(X_valid)
    y_valid = torch.from_numpy(y_valid)
    X_test = torch.from_numpy(X_test)
    y_test  = torch.from_numpy(y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def freeze_forward(BF, forward_net):
    forward_dict = OrderedDict()
    for key in forward_net.state_dict().keys() :
        forward_dict['forward_model.'+key] = forward_net.state_dict().pop(key)

    BF_dict = BF.state_dict()
    BF_dict.update(forward_dict)
    BF.load_state_dict(BF_dict)

    child_count = 0
    for child in BF.children():
        if child_count == 1 :
            for param in child.parameters() :
                param.requires_grad = False
        child_count += 1

    return BF

def plot(file_name):
    plt.rc('font', size = 13)
    plt.rc('axes', titlesize = 13)
    plt.rc('xtick', labelsize = 10)
    plt.rc('ytick', labelsize = 10)

    fig = plt.figure(figsize=(10,15))
    # fig.suptitle('원함수와 역함수 그래프', fontsize=15, fontweight='bold')

    ax1 = fig.add_subplot(3,2,1)
    ax1.scatter(X_test, y_test, alpha=0.2, label='real')
    ax1.scatter(X_test, y_pred, alpha=0.2, label='predicted')
    ax1.legend()
    ax1.set(title='원함수', xlabel='x_axis', ylabel='y_axis')

    ax2 = fig.add_subplot(3,2,2)
    forward_epoch_ = np.arange(0.0, forward_epoch)
    ax2.plot(forward_epoch_, forward_train_loss, label = 'forward network_train_loss')
    ax2.set(xlabel='epochs', ylabel='loss',
           title='Forward Network 학습의 Loss Graph')
    ax2.grid()
    ax2.legend()

    ax3 = fig.add_subplot(3,2,3)
    ax3.scatter(y_test, X_test, alpha=0.2, label='real')
    ax3.scatter(y_test, X_hat, alpha=0.2, label='predicted')
    ax3.legend()
    ax3.set(title='예측된 역함수(x_hat)', xlabel='y_axis', ylabel='x_axis')

    ax4 = fig.add_subplot(3,2,4)
    backward_epoch_ = np.arange(0.0, backward_epoch)
    ax4.plot(backward_epoch_, BF_train_loss, label = 'BF network_train_loss')
    ax4.set(xlabel='epochs', ylabel='loss',
           title='Backward Forward Network 학습의 Loss Graph')
    ax4.grid()
    ax4.legend()

    ax5 = fig.add_subplot(3,2,5)
    ax5.scatter(X_test, y_test, alpha=0.2, label='X_test')
    ax5.scatter(X_hat, y_hat, alpha=0.2, label='X_hat')
    ax5.legend()
    ax5.set(title='예측된 x_hat으로 y_hat 추정', xlabel='x_axis', ylabel='y_axis')


    ax6 = fig.add_subplot(3,2,6)
    ax6.scatter(y_test, y_test, alpha=0.2, label='real')
    ax6.scatter(y_test, y_hat, alpha=0.2, label='predicted')
    ax6.legend()
    ax6.set(title='예측된 y_hat과 실제 y', xlabel='y_axis', ylabel='y_axis')

    plt.subplots_adjust(hspace=0.3)
    plt.savefig('./Figure/split_y/[Figure]{} {} _.png'.format(sc, file_name))
    print('[Figure]{} {}.png has been successfully saved'.format(sc, file_name))
    # plt.show()

def split_y_input_data(y_train, split_num, scale=False) :

    """Quantile에 따라 y_train data를 split하여 input vector를 형성한다."""

    # y_train 데이터를 split 할 기준이 되는 quantile 값들을 구함
    quantile_dict = OrderedDict()
    for x in np.linspace(0, 1, num=split_num, endpoint=True):
        quantile_dict['y_train_quantile_{}'.format(x)] = np.quantile(y_train.cpu().numpy(),x)

    keys = [key for key in quantile_dict.keys()]
    keys_tup = [(keys[i], keys[i+1]) for i in range(len(keys)-1)]
    # print('conditions >>> ',keys_tup)

    y_train_splited = torch.zeros(len(y_train), len(keys_tup))


#     print('Quantile에 따라 split 하는 중..')
    # i는 y_train(y_train_splited)의 각 row index
    for i, data in tqdm(enumerate(y_train)) :
        # j는 y_trained_splited의 column index
        for j, condition in enumerate(keys_tup) :
            if quantile_dict[condition[0]] <= data.cpu().numpy()[0] < quantile_dict[condition[1]] :
                y_train_splited[i][j] = data

    if scale == True :
        # 각 column별로 scaling
        for idx in range(len(keys_tup)) :

            minmaxScaler = MinMaxScaler()
            y_train_splited[:,idx:idx+1] = torch.from_numpy(minmaxScaler.fit_transform(y_train_splited[:, idx:idx+1]))

    return y_train_splited

n_samples = 20000
n_input = 1
n_output = 1
n_forward_hidden = 256
n_forward_layers = 3
n_backward_hidden = 256
n_backward_layers = 1
forward_lr = 0.0001
forward_epoch = 3000
backward_lr = 0.0001
backward_epoch = 12000
split_num = 5
data_adjustment_elements = [0.8]
# data_adjustment_elements = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
# scaler = 'standardScaler'
scaler = 'minmaxScaler'
# scale = False

for scale in [False] :

    if scale :
        sc = '[minmax]'
    else :
        sc = '[no scale]'

    for element in tqdm(data_adjustment_elements) :

        X, y, formula = generate_data(n_samples, element)
        # file_name = formula + "_with {} scaling".format('minmaxScaler')
        file_name = "{} _ split y".format(formula)

        model_path = './model/forward _ {}.tar'.format(file_name)

        X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(X, y, scaler=scaler)

        forward_net = Net(n_feature=n_input, n_hidden=n_forward_hidden, n_output=n_output, layers=n_forward_layers)
        forward_net.cuda()
        forward_net.train()

        optimizer = torch.optim.Adam(forward_net.parameters(), lr=forward_lr)
        X_train = Variable(X_train).cuda()
        y_train = Variable(y_train, requires_grad=False).cuda()
        X_valid = Variable(X_valid, requires_grad=False).cuda()
        y_valid = Variable(y_valid, requires_grad=False).cuda()

        if os.path.isfile(model_path) :
            print('Loading checkpoint "{}"'.format(model_path))
            checkpoint = torch.load(model_path)
            forward_net.load_state_dict(checkpoint['forward_state_dict'])
            forward_net.to(torch.device("cuda"))
            forward_train_loss  = checkpoint['forward_train_loss']
            forward_valid_loss = checkpoint['forward_valid_loss']
        else :
            print('Training Forward Network ..', end='\n\n')
            forward_result = train(forward_net, X_train, y_train, X_valid, y_valid, optimizer, forward_epoch)
            forward_train_loss = forward_result[0]
            forward_valid_loss = forward_result[1]

            torch.save({
            'forward_state_dict' : forward_net.state_dict(),
            'forward_train_loss' : forward_train_loss,
            'forward_valid_loss' : forward_valid_loss,
            }, model_path)

        forward_net.eval()
        y_pred = forward_net(X_test.cuda()).cpu().data.numpy()
        print("r2_score(y_test, y_pred) : ", r2_score(y_test, y_pred), end='\n\n')

        print('Processing y_input_data ..', end='\n\n')
        y_train_ = split_y_input_data(y_train, split_num, scale=scale)
        y_valid_ = split_y_input_data(y_valid, split_num, scale=scale)
        y_test_ = split_y_input_data(y_test, split_num, scale=scale)

        n_input_backward = y_train_.shape[1]
        backward_net = Net_(n_feature=n_input_backward, n_hidden=n_backward_hidden, n_output=n_output, layers=n_backward_layers)

        print('Building Backward-Forward Network ..', end='\n\n')
        BF = BackwardForward(backward_net, forward_net)
        BF.cuda()
        BF = freeze_forward(BF, forward_net)

        y_train_input = Variable(y_train_).cuda()
        y_train_label = Variable(y_train, requires_grad=False).cuda()
        y_valid_input = Variable(y_valid_, requires_grad=False).cuda()
        y_valid_label = Variable(y_valid, requires_grad=False).cuda()
        optimizer_bf = torch.optim.RMSprop(BF.parameters(), lr=backward_lr)

        BF.train()
        print('Training Backward Network ..', end='\n\n')
        BF_result = train(BF, y_train_input, y_train_label, y_valid_input, y_valid_label, optimizer_bf, backward_epoch)
        BF_train_loss = BF_result[0]
        BF_valid_loss = BF_result[1]

        BF.eval()
        X_hat = BF.backward_model(y_test_.cuda()).cpu().data.numpy()
        y_hat = BF(y_test_.cuda()).cpu().data.numpy()

        print('r2_score(y_test, y_hat) : ', r2_score(y_test, y_hat), end='\n\n')

        plot(file_name)

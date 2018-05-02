import torch
from torch.autograd import Variable
import torch.nn as nn
import resnet_conv


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        # print(recurrent.size())
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNNCLS(nn.Module):

    def __init__(self, n_class, n_hidden, n_rnn=2, head_net='res50'):
        '''
        # 用于判断按顺序输入的三张图像是否符合要求，每张图像的宽高为imgH，
        #输入为b*3,3,h,w

        :param img_h: 图像宽高
        :param n_class: 分类数
        :param n_hidden: lstm的隐层个数
        :param headnet: 特征提取网络，目前支持'res50','rs101','res152'三种
        '''
        super(CRNNCLS, self).__init__()
        assert n_rnn > 0, 'the layers of rnn must greater than 0'



        if head_net == 'res101':
            self.cnn = resnet_conv.resnet101()
        elif head_net == 'res152':
            self.cnn = resnet_conv.resnet152()
        else:
            self.cnn = resnet_conv.resnet50()      #[b*3, 2048, 10, 32]

        self.pool = nn.AdaptiveMaxPool2d(1)    #[b*3, 2048, 3, 3]
        # self.squeeze = nn.Conv2d(2048, 256, kernel_size=1) #压缩维度
        # nn.init.xavier_uniform(self.squeeze.weight)

        # self.rnn = nn.Sequential()
        last_input_dim = 2048
        self.lstm = nn.LSTM(last_input_dim, n_hidden, num_layers=n_rnn)
        # if n_rnn > 2:
        #     for i in range(1, n_rnn):
        #         self.rnn.add_module('LSTM{}'.format(i), BidirectionalLSTM(last_input_dim, n_hidden, n_hidden))
        #         last_input_dim = n_hidden
        # elif n_rnn == 2:
        #     self.rnn.add_module('LSTM1', BidirectionalLSTM(last_input_dim, n_hidden, n_hidden))
        #     last_input_dim = n_hidden
        # self.rnn.add_module('LSTM{}'.format(n_rnn), nn.LSTM(last_input_dim, n_hidden))

        self.out_cls1 = nn.Linear(n_hidden, n_class)
        self.out_cls2 = nn.Linear(n_hidden, n_class)
        self.out_cls3 = nn.Linear(n_hidden, n_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.interstate = nn.Linear(last_input_dim*3, 512*3)
        # self.activate = nn.ReLU()
        # self.clsnew = nn.Linear(512*3, n_class)
    #     self.hidden = self.init_hidden()
    #
        #fix block
        for p in self.cnn.conv1.parameters():
            p.requires_grad=False
        for p in self.cnn.bn1.parameters():
            p.requires_grad=False
        for p in self.cnn.layer1.parameters():
            p.requires_grad=False
        # for p in self.cnn.layer2.parameters():
        #     p.requires_grad=False
        # for p in self.cnn.layer3.parameters():
        #     p.requires_grad=False
        # for p in self.cnn.layer4.parameters():
        #     p.requires_grad=False



    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # print('conv = self.cnn(input) size:', conv.size())
        conv = self.pool(conv)    #[batch*3, 2048, 1, 1]
        # conv = self.squeeze(conv)
        b, c, h, w = conv.size()
        # print('conv = self.pool_3step(conv) size:', conv.size())

        # conv = conv.permute(0, 2, 3, 1)  # [b, h*w, c]注意permute后内存是不变的，只是改变了tensor的stride
        # conv = conv.contiguous().view(b, -1)
        conv = conv.view(b, -1)
        conv_split = torch.split(conv, 3, dim=0)   #序列长度为w*h*3 #rnn的batch为b/3
        rnn_feature = torch.stack(conv_split, dim=1)
        # print('rnn_feature = torch.stack(conv_split, dim=1):', rnn_feature.size())

        # conv = conv.squeeze(2)
        # conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features input:setp,batch,feature_size
        output, _ = self.lstm(rnn_feature)
        # print(output)
        # print(output.size())
        # print(output[2].size())
        res1 = self.out_cls1(output[0])
        res2 = self.out_cls2(output[1])
        res3 = self.out_cls3(output[2])

        # conv = conv.view(int(b/3), -1)
        # out = self.interstate(conv)
        # out = self.activate(out)
        # res = self.clsnew(out)

        return res1, res2, res3

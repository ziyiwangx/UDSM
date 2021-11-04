import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable, Function


class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = ~(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output*mask
        return grad_output, None, None


def hard_sigmoid(x):
    y = (x+1.)/2.
    y[y>1] = 1
    y[y<0] = 0
    return y


def binary_tanh_unit(x):
    # round3 = Elemwise(round3_scalar)
    y = hard_sigmoid(x)
    # pdb.set_trace()
    out = 2.*Round3.apply(y)-1.
    return out


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, batch_size, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False,
                            dropout=0.2, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.batch_size = batch_size

        # initialize weights
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()

        # forward propagate lstm
        out, hidden_cell = self.lstm(x, (h0, c0))  # out: tensor of shape (seq_length, batch_size, hidden_size)

        # return out[:, -1, :].unsqueeze(1)
        return out[-1, :, :].unsqueeze(0), hidden_cell


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, batch_size, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=False,
                            dropout=0.2, bidirectional=bidirectional)
        self.batch_size = batch_size
        self.linear = nn.Linear(in_features=output_size, out_features=hidden_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    '''def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, self.batch_size, self.output_size).cuda()
        c0 = torch.zeros(self.num_layers, self.batch_size, self.output_size).cuda()

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (seq_length, batch_size, hidden_size)

        return out'''

    def forward(self, input_seq, hidden_cell):
        input_seq = self.linear(input_seq)
        output, hidden_cell = self.lstm(input_seq, hidden_cell)

        return output, hidden_cell


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, batch_size, bidirectional)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, batch_size, bidirectional)
        self.input_size = input_size
        self.batch_size = batch_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.embedding1 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        sequence_length = 20

        # 256的hidden输入到decoder， decoder接下来的输入是上一个lstm的输出
        encoded_x, (h, c) = self.encoder(x)

        hidden_cell = (self.fc(h), self.fc(c))
        # print('hc', h.size(), c.size())

        output = torch.zeros(size=x.shape, dtype=torch.float).cuda()
        input_decoder = self.fc(encoded_x)
        # print(input_decoder)
        input_decoder_binary = binary_tanh_unit(input_decoder)    # decoder 二值输入
        input_decoder = input_decoder_binary    # todo decoder 二值输入
        # print(input_decoder)
        # hidden_cell = self.fc(encoded_x).unsqueeze(0)
        for i in range(sequence_length-1,-1,-1):   # (19,,,,0)
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output[i, :, :] = output_decoder[0, :, :]   # (1,64,32)
            # print('output', output.size(), output_decoder.size(), 'hidden_cell', hidden_cell[0].size())
        # print('output', output)
        return output, input_decoder_binary.squeeze()

import torch
import torch.nn as nn
from torch.autograd import Variable as V

DROPOUT = 0.1


class Unet_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, n_layers=1):
        super(Unet_LSTM, self).__init__()

        self.down1 = self.conv_stage(1, 8)
        self.down2 = self.conv_stage(8, 16)
        self.down3 = self.conv_stage(16, 32)
        self.down4 = self.conv_stage(32, 64)
        self.down5 = self.conv_stage(64, 128)

        self.lstm1 = nn.LSTM(input_size=248, hidden_size=128, num_layers=n_layers, dropout=DROPOUT,
                             batch_first=True)
        self.conv_d1 = nn.Sequential(
            nn.Conv1d(256, 128, 1, 1, 0),
        )

        self.center = self.conv_stage(128, 256)

        self.lstm2 = nn.LSTM(input_size=496, hidden_size=16, num_layers=n_layers, dropout=DROPOUT,
                             batch_first=True)

        self.up5 = self.conv_stage(256, 128)
        self.up4 = self.conv_stage(256, 64)
        self.up3 = self.conv_stage(128, 32)
        self.up2 = self.conv_stage(64, 16)
        self.up1 = self.conv_stage(32, 8)

        self.conv_d2 = nn.Sequential(
            nn.Conv1d(32, 8, 1, 1, 0),
        )

        self.conv_last = nn.Sequential(
            nn.Conv1d(16, 1, 3, 1, 1),
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(),
                nn.Conv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                nn.Conv1d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):

        conv1_out = self.down1(x)
        conv2_out = self.down2(conv1_out)
        conv3_out = self.down3(conv2_out)
        conv4_out = self.down4(conv3_out)
        conv5_out = self.down5(conv4_out)

        lstm_in1 = torch.cat((conv1_out, conv2_out, conv3_out, conv4_out, conv5_out), 1)
        lstm_in1 = lstm_in1.transpose(2, 1)
        lstm_out1, _ = self.lstm1(lstm_in1)
        lstm_out1 = lstm_out1.transpose(2, 1)
        center_out = self.center(self.conv_d1(torch.cat((conv5_out, lstm_out1), 1)))

        out1 = self.up5(center_out)
        out2 = self.up4(torch.cat((out1, conv5_out), dim=1))
        out3 = self.up3(torch.cat((out2, conv4_out), dim=1))
        out4 = self.up2(torch.cat((out3, conv3_out), dim=1))


        lstm_in2 = torch.cat((center_out, out1, out2, out3, out4), 1)
        lstm_in2 = lstm_in2.transpose(2, 1)
        lstm_out2, _ = self.lstm2(lstm_in2)
        lstm_out2 = lstm_out2.transpose(2, 1)
        lstm_out2 = self.conv_d2(torch.cat((out4, lstm_out2), 1))

        out = self.conv_last(torch.cat((lstm_out2, conv1_out), dim=1))

        return out

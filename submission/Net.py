import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn




class Net(nn.Module):
    def __init__(self, hidden_size=256):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.modules = []

        input_size = 40
        lstm_hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size,
                       hidden_size=lstm_hidden_size,
                       num_layers=3,
                       bidirectional=True,
                            batch_first=True)
        self.modules.append(self.lstm)

        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=47)
        self.modules.append(self.linear)
        self.net = nn.Sequential(*self.modules)

    def forward(self, frames):
        lstm_out, _ = self.lstm(frames)
        lstm_out, _ = rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # print("LSTM unpacked output shape: " + repr(lstm_out.size()))
        linear_out = self.linear(lstm_out)
        # print("Linear output shape: " + repr(linear_out.size()))
        out = F.log_softmax(linear_out, dim=2)
        return out
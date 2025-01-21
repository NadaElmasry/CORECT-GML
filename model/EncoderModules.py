import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqTransfomer(nn.Module):
    def __init__(self, input_size, h_dim, args):
        super(SeqTransfomer, self).__init__()

        self.input_size = input_size
        self.nhead = self.calculate_nhead(input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=self.nhead,
            dropout=args.drop_rate,  
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, args.encoder_nlayers)

        if input_size != h_dim:
            self.transformer_out = torch.nn.Linear(input_size, h_dim, bias=True)
        else:
            self.transformer_out = nn.Identity()
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.transformer_out(x)
        return x

    @staticmethod
    def calculate_nhead(input_size):
        # Find the largest factor of input_size in the range [7, 15]
        for h in range(15, 6, -1):
            if input_size % h == 0:
                return h
        return 1  # Default to 1 if no suitable factor is found



class FC_with_PE(nn.Module):
    def __init__(self, input_size, h_dim, args):
        super(FC_with_PE, self).__init__()

        self.input_size = input_size
        self.hidden_dim = h_dim
        self.args = args

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.fc(x)
        return x



class LSTM_Layer(nn.Module):
    def __init__(self, input_size, hidden_dim, args):
        super(LSTM_Layer, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_dim // 2,
                            bidirectional=True,
                            num_layers=args.encoder_nlayers,
                            batch_first=True)
        
        self.dropout = nn.Dropout(args.drop_rate)
        
    def forward(self, x, lengths):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.hidden_dim // 2).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.hidden_dim // 2).to(x.device)
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed, (h0, c0))
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)  
        return out
    
class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_enc', self._get_positional_encoding(max_len, d_model))

    def _get_positional_encoding(self, max_len, d_model):
        position = torch.linspace(0, max_len - 1, steps=max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        return pos_enc.unsqueeze(0).transpose(0,1)

    def forward(self, x):
        pos_enc = self.pos_enc[:, :x.size(0), :].to(x.device) 
        x = x + pos_enc
        return self.dropout(x)

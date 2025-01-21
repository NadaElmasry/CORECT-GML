import torch
import torch.nn as nn

from .EncoderModules import SeqTransfomer, LSTM_Layer

class UnimodalEncoder(nn.Module):
    def __init__(self, a_dim, t_dim, v_dim, h_dim, args):
        super(UnimodalEncoder, self).__init__()
        self.hidden_dim = h_dim
        self.device = args.device
        self.rnn = args.rnn      
        self.a_dim = a_dim
        self.t_dim = t_dim
        self.v_dim = v_dim
       
        self.audio_encoder = self.gen_encoder(self.a_dim, self.hidden_dim)
        self.text_encoder = self.get_text_encoder(t_dim, h_dim, args)
        self.vision_encoder = self.gen_encoder(self.v_dim, self.hidden_dim)
        

    def gen_encoder(self, vec_dim, hidden_dim):
        return  nn.Sequential(  nn.Linear(vec_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                )


    def get_text_encoder(self, t_dim, h_dim, args):
        if self.rnn == "transformer":
            return SeqTransfomer(t_dim, h_dim, args)
        elif self.rnn == "ffn":
            return self.gen_encoder(t_dim, h_dim)
        elif self.rnn == "lstm":
            return LSTM_Layer(t_dim, h_dim, args)
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn}")

    def forward(self, a, t, v, lengths=None):
        a_out = self.audio_encoder(a) if a is not None else None
        t_out = self.text_encoder(t, lengths) if t is not None and self.rnn == "lstm" else self.text_encoder(t)
        v_out = self.vision_encoder(v) if v is not None else None

        return t_out, a_out, v_out

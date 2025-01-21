import torch
import torch.nn as nn
import torch.nn.functional as F

from .Classifier import Classifier
from .UnimodalEncoder import UnimodalEncoder
from .CrossModalNet import CrossmodalNet
from .GraphModel import GraphModel
from utils import multi_concat, feature_packing
import utils

log = utils.get_logger()

class CORECT(nn.Module):
    def __init__(self, args):
        super(CORECT, self).__init__()
        self.args = args
        self.modalities = args.modalities
        self.n_modals = len(self.modalities)
        self.use_speaker = args.use_speaker
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        self.log = {}
        g_dim = args.hidden_size
        h_dim = args.hidden_size
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
                   }


        tag_size = len(dataset_label_dict[args.dataset])
        self.n_speakers = 2
        print(f"Using {args.dataset} Dataset, Number of Speakers = {self.n_speakers}")



        ic_dim = 0
        if not args.no_gnn:
            ic_dim = h_dim * self.n_modals

            if not args.use_graph_transformer and (args.gcn_conv in ["gat_gcn","gcn_gat"]):
                ic_dim = ic_dim * 2

            if args.use_graph_transformer:
                ic_dim *= args.graph_transformer_nheads
        
        if args.use_crossmodal and self.n_modals > 1:
            ic_dim += h_dim * self.n_modals * (self.n_modals - 1)

        if self.args.no_gnn and (not self.args.use_crossmodal or self.n_modals == 1):
            ic_dim = h_dim * self.n_modals
        a_dim, t_dim, v_dim = [args.dataset_embedding_dims[args.dataset][m] for m in ['a', 't', 'v']]
                

        self.encoder = UnimodalEncoder(a_dim, t_dim, v_dim, g_dim, args)
        self.speaker_embedding = nn.Embedding(self.n_speakers, g_dim)

        if not args.no_gnn:
            self.graph_model = GraphModel(g_dim, h_dim, h_dim, args.device, args)
            print("CORECT Model is using GNN")
        if args.use_crossmodal and self.n_modals > 1:
            self.crossmodal = CrossmodalNet(g_dim, args)
            print("CORECT Model is using CrossModalNet")
        if self.n_modals == 1:
            print("CORECT only works with multiple modalities, a single modality or less was entered")
        self.clf = Classifier(ic_dim, h_dim, tag_size, args)

    def represent(self, data):
        a = data.get('audio_tensor')
        t = data.get('text_tensor')
        v = data.get('visual_tensor')

        a, t, v = self.encoder(a, t, v, data['text_len_tensor'])

        if self.use_speaker:
            emb = self.speaker_embedding(data['speaker_tensor'])
            a = a + emb if a is not None else None
            t = t + emb if t is not None else None
            v = v + emb if v is not None else None

        multimodal_features = [x for x in [a, t, v] if x is not None]
        out_encode = feature_packing(multimodal_features, data['text_len_tensor'])
        out_encode = multi_concat(out_encode, data['text_len_tensor'], self.n_modals)

        out = []
        if not self.args.no_gnn:
            out_graph = self.graph_model(multimodal_features, data['text_len_tensor'])
            out.append(out_graph)

        if self.args.use_crossmodal and self.n_modals > 1:
            out_cr = self.crossmodal(multimodal_features)
            out_cr = out_cr.permute(1, 0, 2)
            lengths = data['text_len_tensor']
            cr_feat = torch.cat([out_cr[j, :lengths[j].item()] for j in range(lengths.size(0))], dim=0).to(self.device)
            out.append(cr_feat)

        out = out_encode if self.args.no_gnn and (not self.args.use_crossmodal or self.n_modals == 1) else torch.cat(out, dim=-1)
        return out

    def forward(self, data):
        graph_out = self.represent(data)
        out = self.clf(graph_out, data["text_len_tensor"])
        return out

    def get_loss(self, data):
        graph_out = self.represent(data)
        loss = self.clf.get_loss(graph_out, data["label_tensor"], data["text_len_tensor"])
        return loss

    def get_log(self):
        return self.log

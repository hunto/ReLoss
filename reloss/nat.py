from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor
from fairseq import metrics, utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.modules import TransformerEncoderLayer, PositionalEmbedding
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion
from fairseq import optim
from torch.hub import load_state_dict_from_url


class DNNLoss(nn.Module):
    def __init__(self, args, vocab_size, padding_idx, fn=nn.ELU, pretrained=True):
        super(DNNLoss, self).__init__()
        self.logits_layer = nn.Sequential(
            nn.Linear(1, args.decoder_embed_dim),
            fn(),
            nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim),
            fn(),
        )
        new_args = deepcopy(args)
        new_args.dropout = 0.1
        new_args.activation_fn = 'elu'
        self.ce = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.pe = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, padding_idx, learned=True)
        self.emb_drop = nn.Dropout(new_args.dropout)
        self.block = nn.ModuleList([TransformerEncoderLayer(new_args) for _ in range(2)])
        self.loss_layer = nn.Sequential(
            nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim),
            nn.ELU(),
            nn.Linear(args.decoder_embed_dim, 1),
        )
        if pretrained:
            ckpt_url = 'https://github.com/hunto/ReLoss/releases/download/v1.0.0/reloss_nat.ckpt'
            print(f'Load checkpoint of ReLoss from url: {ckpt_url}')
            state_dict = load_state_dict_from_url(ckpt_url, map_location='cpu')
            self.load_state_dict(state_dict)


    def forward(self, model, logits, tgt, tgt_non_pad_mask):
        batch_size, seq_len, vocab_size = logits.size()
        pos_logits = torch.gather(logits, 2, tgt.unsqueeze(-1))
        pos_logits = pos_logits * tgt_non_pad_mask.unsqueeze(-1)
        pos_hidden = self.logits_layer(pos_logits)
        pos_hidden = self.emb_drop(pos_hidden + self.pe(tgt_non_pad_mask))
        pos_hidden = self.block[0](pos_hidden.transpose(0, 1), ~tgt_non_pad_mask).transpose(0, 1)
        pos_hidden = self.block[1](pos_hidden.transpose(0, 1), ~tgt_non_pad_mask).transpose(0, 1)
        loss = self.loss_layer(pos_hidden).squeeze(-1) * tgt_non_pad_mask
        loss = loss.sum(1) / (tgt_non_pad_mask.sum(1) + 1e-8)
        return loss.abs().mean().unsqueeze(0)


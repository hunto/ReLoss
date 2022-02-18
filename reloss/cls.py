import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url


class ReLoss(nn.Module):

    def __init__(self, hidden_dim=128, pretrained=True):
        super(ReLoss, self).__init__()
        self.logits_fcs = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.loss = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # load pretrained weights
        if pretrained:
            ckpt_url = 'https://github.com/hunto/ReLoss/releases/download/v1.0.0/reloss_cls.ckpt'
            print(f'Load checkpoint of ReLoss from url: {ckpt_url}')
            state_dict = load_state_dict_from_url(ckpt_url, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pos_probs = torch.gather(probs, 1, targets.unsqueeze(-1))
        hidden = self.logits_fcs(pos_probs)
        loss = self.loss(hidden).abs().mean()
        return loss

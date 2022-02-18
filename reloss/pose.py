import torch.nn as nn

from torch.hub import load_state_dict_from_url


class ReLoss(nn.Module):

    def __init__(self, hidden_dim=8, heatmap_size=(64, 48), pretrained=True):
        super(ReLoss, self).__init__()
        self.logits_fcs = nn.Sequential(
            nn.Linear(heatmap_size[0] * heatmap_size[1], hidden_dim), 
            nn.ELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # load pretrained weights
        if pretrained:
            if heatmap_size == (64, 48):
                ckpt_url = 'https://github.com/hunto/ReLoss/releases/download/v1.0.0/reloss_pose_256x192.ckpt'
            elif heatmap_size == (96, 72):
                ckpt_url = 'https://github.com/hunto/ReLoss/releases/download/v1.0.0/reloss_pose_384x288.ckpt'
            else:
                raise RuntimeError(
                    f'Incompatible heatmap size {heatmap_size}.')
            print(f'Load checkpoint of ReLoss from url: {ckpt_url}')
            state_dict = load_state_dict_from_url(ckpt_url, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, output, target, mask):
        '''
        args:
            @output: [N, K, H, W]
            @target: [N, K, H, W]
            @mask: [N, K]
        return:
            @loss: [1]
        '''
        N, K, H, W = output.shape
        mask = mask.view(N, K, 1, 1)
        output = output * mask
        target = target * mask
        mse = ((output - target)**2).view(N, K, -1)
        attn = self.logits_fcs(mse)  # [N, K, 1]
        loss = (mse * attn).mean()
        return loss

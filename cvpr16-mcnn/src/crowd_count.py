import torch
import torch.nn as nn
from model import MCNN


class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        self.DME = MCNN()
        if torch.cuda.is_available():
            self.DME.cuda()
        self.loss_fn = nn.MSELoss()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_density=None):
        image = torch.from_numpy(img)
        if torch.cuda.is_available():
            image = image.cuda()
        density_map = self.DME(image)

        if self.training:
            gt_density = torch.from_numpy(gt_density)
            if torch.cuda.is_available():
                gt_density = gt_density.cuda()
            self.loss_mse = self.loss_fn(density_map, gt_density)
        return density_map

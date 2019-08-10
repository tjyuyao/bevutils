import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from bevutils.datasets import SimulatedDataSet
from bevutils.layers import PerspectiveTransformerLayer
from bevutils.models import DATA_LOADERS, MODELS, LOSSES, METRICS, BaseDataLoader, BaseModel
from bevutils.trainer.utils import parse_device


@MODELS.register
class PerspectiveTransformerNetwork(BaseModel):
    def __init__(self, bv_size, pv_size, intrinsics, translate_z = -10.0, rotation_order='xyz'):
        super().__init__()
        self.rotation_order = rotation_order
        # layers
        self.ptl0 = PerspectiveTransformerLayer(bv_size, pv_size, intrinsics, translate_z, rotation_order)
        self.encoder = nn.Sequential(
            nn.Conv2d(  1,  32, 5, stride=1, padding=2), nn.SELU(True),
            nn.Conv2d( 32,  64, 3, stride=1, padding=1), nn.SELU(True),
            nn.Conv2d( 64, 128, 3, stride=2, padding=1), nn.SELU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SELU(True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SELU(True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*bv_size[0]//8, 1024), nn.SELU(True),
            nn.Linear(1024, 128), nn.SELU(True),
            nn.Linear(128, len(rotation_order)), nn.Tanh()
        )
        self.ptl1 = PerspectiveTransformerLayer(bv_size, pv_size, intrinsics, translate_z, rotation_order)

    def forward(self, input):
        pv, rx0, ry0, rz0 = input
        angles_base = {'rx':rx0, 'ry':ry0, 'rz':rz0}
        bv0 = self.ptl0(pv, rx0, ry0, rz0)
        feat = self.encoder(bv0)
        rowfeat = torch.sum(F.softmax(feat, dim=-1) * feat, dim=-1)
        rowfeat = rowfeat.view((rowfeat.shape[0], -1))
        angles_delta = self.fc1(rowfeat)
        angles = {}
        for i, x_str in enumerate(self.rotation_order):
            rx_str = 'r' + x_str
            angles[rx_str] = angles_base[rx_str] + angles_delta[:, i]
        x = self.ptl1(pv, **angles)
        return x, angles

@MODELS.register
class PerspectiveTransformerNetwork_DensePred(BaseModel):
    def __init__(self, bv_size, pv_size, intrinsics, translate_z = -10.0, rotation_order='xyz'):
        super().__init__()
        self.rotation_order = rotation_order
        # layers
        self.ptl0 = PerspectiveTransformerLayer(bv_size, pv_size, intrinsics, translate_z, rotation_order)
        self.encoder = nn.Sequential(
            nn.Conv2d(  1,  32, 5, stride=1, padding=2), nn.SELU(True),
            nn.Conv2d( 32,  64, 3, stride=1, padding=1), nn.SELU(True),
            nn.Conv2d( 64, 128, 3, stride=2, padding=1), nn.SELU(True),
        )
        self.angle_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SELU(True),
            nn.Conv2d(128, 1, 3, stride=2, padding=1), 
        )
        self.attention_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SELU(True),
            nn.Conv2d(128, 1, 3, stride=2, padding=1), nn.SELU(True)
        )
        self.ptl1 = PerspectiveTransformerLayer(bv_size, pv_size, intrinsics, translate_z, rotation_order)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)


    def forward(self, input):
        pv, rx0, ry0, rz0 = input
        angles_base = {'rx':rx0, 'ry':ry0, 'rz':rz0}
        bv0 = self.ptl0(pv, rx0, ry0, rz0)
        feat = self.encoder(bv0)
        angles_delta = self.angle_head(feat) * 0.01
        attention = self.attention_head(feat)
        B, C, H, W = attention.shape
        angles_delta = angles_delta.view((B, C, H*W))
        attention = F.softmax(attention.view((B, C, H*W)), dim=-1)
        angles_delta = torch.sum(angles_delta * attention, dim=-1)
        angles = {}
        for i, x_str in enumerate(self.rotation_order):
            rx_str = 'r' + x_str
            angles[rx_str] = angles_base[rx_str] + angles_delta[:, i]
        x = self.ptl1(pv, **angles)
        return x, angles



@DATA_LOADERS.register
class SimulatedDataLoader(BaseDataLoader): # FIXME narrow random range
    def __init__(self, bv_size, pv_size, intrinsics, translate_z, rotation_order,
        length, batch_size, shuffle=True, validation_split=0.0, num_workers=1, min_rx=1.65, max_rx=1.77):
        self.dataset = SimulatedDataSet(
            length=length, bv_size=bv_size, pv_size=pv_size, intrinsics=intrinsics,
            translate_z = translate_z, rotation_order=rotation_order, to_tensor=True,
            rx = lambda: random.uniform(min_rx, max_rx),
            ry = lambda: 0.0
            )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


@LOSSES.register
class PerspectiveLoss():
    def __init__(self, w_bv):
        self.w_bv = w_bv
    def __call__(self, output, target):
        bv_gt, rx, ry, rz = target
        angle_gt = {'rx' : rx, 'ry': ry, 'rz':rz}
        bv, angle = output
        loss = dict()
        loss['loss_bv'] = torch.mean((bv-bv_gt)**2) * self.w_bv
        loss['loss_ang'] = torch.zeros(1, device=rx.device)
        for k in angle:
            loss['loss_ang'] += torch.mean((angle[k] - angle_gt[k])**2)
        loss['loss'] = torch.mean(loss['loss_bv'] + loss['loss_ang'])
        return loss

@METRICS.register
def rmse_rx(output, target):
    bv_gt, rx, ry, rz = target
    bv, angle = output
    return torch.mean(torch.sqrt((angle['rx'] - rx)**2))


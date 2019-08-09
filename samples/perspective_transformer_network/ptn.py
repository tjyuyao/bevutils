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
            angles[rx_str] = angles_base[rx_str] + angles_delta[i]
        x = self.ptl1(pv, **angles)
        return x, angles


@DATA_LOADERS.register
class SimulatedDataLoader(BaseDataLoader): # FIXME narrow random range
    def __init__(self, bv_size, pv_size, intrinsics, translate_z, rotation_order,
        length, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = SimulatedDataSet(
            length=length, bv_size=bv_size, pv_size=pv_size, intrinsics=intrinsics,
            translate_z = translate_z, rotation_order=rotation_order, to_tensor=True)
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
            loss['loss_ang'] += (angle[k] - angle_gt[k])**2
        loss['loss'] = loss['loss_bv'] + loss['loss_ang']
        return loss
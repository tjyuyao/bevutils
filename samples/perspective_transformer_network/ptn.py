import torch
import torch.nn as nn
import torch.nn.functional as F

from bevutils.datasets import SimulatedDataSet
from bevutils.layers import PerspectiveTransformerLayer
from bevutils.models import DATA_LOADERS, MODELS, LOSSES, METRICS, BaseDataLoader, BaseModel
from bevutils.trainer.utils import parse_device


@MODELS.register
class PerspectiveTransformerNetwork(BaseModel):
    def __init__(self, bv_size, pv_size, intrinsics, translate_z = 0.0, rotation_order='xyz'):
        super().__init__()
        self.fc_len = pv_size[0] * pv_size[1] // 4 // 4 * 10
        self.rotation_order = rotation_order
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_len, 50)
        self.fc2 = nn.Linear(50, len(self.rotation_order))
        self.ptl = PerspectiveTransformerLayer(bv_size, pv_size, intrinsics, translate_z, rotation_order)

    def forward(self, x):
        raw = x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.fc_len)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        angles = self.fc2(x)
        angles = {'r'+self.rotation_order[i] : angles[:, i] for i in range(angles.shape[1])}
        x = self.ptl(raw, **angles)
        return x, angles

@DATA_LOADERS.register
class SimulatedOverfitDataLoader(BaseDataLoader):
    def __init__(self, bv_size, pv_size, intrinsics, translate_z, rotation_order,
        length, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = list(SimulatedDataSet(
            length=length, bv_size=bv_size, pv_size=pv_size, intrinsics=intrinsics,
            translate_z = translate_z, rotation_order=rotation_order, to_tensor=True))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

@LOSSES.register
class PerspectiveLoss():
    def __init__(self, w):
        self.w = w
    def __call__(self, output, target):
        bv_gt, rx, ry, rz = target
        angle_gt = {'rx' : rx, 'ry': ry, 'rz':rz}
        bv, angle = output
        loss_bv = torch.mean((bv-bv_gt)**2)
        loss_ang = torch.zeros(1, device=rx.device)
        for k in angle:
            loss_ang += (angle[k] - angle_gt[k])**2
        return loss_ang + loss_bv * self.w
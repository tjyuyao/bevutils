import random
import cv2
import numpy as np
from numpy.linalg import inv
import torch
import torch.utils.data
from ..functional import epipolar as E

class SimulatedDataSet(torch.utils.data.Dataset):
    def __init__(self, 
                 bv_size=(1000, 500),
                 pv_size=(480, 640),
                 intrinsics=[
                     [437.2959,        0, 319.9692], 
                     [       0, 408.1196, 239.8203], 
                     [       0,        0,        1]], 
                 rx=lambda : random.uniform(1.4, 1.8),
                 ry=lambda : random.uniform(-0.04, 0.04),
                 rz=lambda : 0.0,
                 rotation_order='xyz',
                 translate_z=-10,
                 line_width=lambda : random.randint(10, 20),
                 line_counts=lambda : random.randint(1, 2),
                 circ_counts=lambda : 1,
                 bg=lambda : 0,
                 fg=lambda : 255,
                 length=100,
                 to_tensor=False,
                 dtype=torch.float32
        ):
        self.K = np.array(intrinsics, dtype='float')
        self.bv_size = bv_size # (H, W)
        self.pv_size = (pv_size[1], pv_size[0]) # (W, H)
        self.length = length
        self.rx, self.ry, self.rz = rx, ry, rz
        self.rot_order = rotation_order
        self.tz = translate_z
        self.line_width = line_width
        self.line_counts = line_counts
        self.circ_counts = circ_counts
        self.bg, self.fg = bg, fg
        self.bv_pivot = [bv_size[1]/2, bv_size[0], 1]
        self.pv_pivot = [self.K[1,2], self.K[0,2]*2, 1]
        self.to_tensor = to_tensor
        self.device = torch.device("cpu")
        self.dtype = dtype

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError
        bv = np.full(shape=self.bv_size, fill_value=self.bg(), dtype=np.uint8)
        self.__draw_random_lines_on(bv, self.line_counts())
        self.__draw_random_circles_on(bv, self.circ_counts())
        rx, ry, rz = self.rx(), self.ry(), self.rz()
        R = E.numpy.make_rotation_matrix(rx, ry, rz, self.rot_order)
        H = E.numpy.make_constrained_homography(R, self.tz, self.K, bv_pivot=self.bv_pivot, pv_pivot=self.pv_pivot)
        pv = cv2.warpPerspective(bv, inv(H), self.pv_size)
        if self.to_tensor:
            [bv, pv, rx, ry, rz] = [torch.tensor(i, device=self.device, dtype=self.dtype) 
                                    for i in [bv[np.newaxis, :, :], pv[np.newaxis, :, :], rx, ry, rz]]
        return pv, (bv, rx, ry, rz)

    def __draw_random_lines_on(self, bv, num=1):
        H, W = self.bv_size
        for i in range(num):
            pt1 = (random.randint(0, W), H)
            pt2 = (random.randint(0, W), random.randint(0, H/2))
            cv2.line(bv, pt1, pt2, self.fg(), self.line_width())

    def __draw_random_circles_on(self, view, num=1):
        H, W = self.bv_size
        for i in range(num):
            cxrange = list(range(-W, 0)) + list(range(W, 2*W))
            center = (random.choice(cxrange), random.randint(3*H/4, H))
            radius = random.randint(W/2, H)
            cv2.circle(view, center, radius, self.fg(), self.line_width())

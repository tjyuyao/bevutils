import random
import cv2
import numpy as np
from numpy.linalg import inv
import torch
import torch.utils.data
from ..functional import make_rot_mat, make_homography

class SimulatedDataSet(torch.utils.data.Dataset):
    def __init__(self, 
                 bvsize=(1000, 500),
                 pvsize=(480, 640),
                 intrinsics=[
                     [437.2959,        0, 319.9692], 
                     [       0, 408.1196, 239.8203], 
                     [       0,        0,        1]], 
                 rx=lambda : random.uniform(1.4, 1.8),
                 ry=lambda : random.uniform(-0.04, 0.04),
                 rz=lambda : 0.0,
                 tz=-10,
                 line_width=lambda : random.randint(10, 20),
                 line_counts=lambda : random.randint(1, 2),
                 circ_counts=lambda : 1,
                 bg=lambda : 0,
                 fg=lambda : 255,
                 length=100,
                 device=None,
                 dtype=torch.float32
        ):
        self.K = torch.tensor(intrinsics, dtype=dtype, device=device)
        self.bvsize = bvsize # (H, W)
        self.pvsize = (pvsize[1], pvsize[0]) # (W, H)
        self.length = length
        self.rx, self.ry, self.rz = rx, ry, rz
        self.tz = torch.tensor([tz], dtype=dtype, device=device)
        self.line_width = line_width
        self.line_counts = line_counts
        self.circ_counts = circ_counts
        self.bg, self.fg = bg, fg
        self.device=device
        self.dtype=dtype
        self.bv_pivot = torch.tensor([bvsize[1]/2, bvsize[0], 1], dtype=self.dtype, device=self.device).view(3, 1)
        self.pv_pivot = torch.tensor([self.K[1,2], self.K[0,2]*2, 1], dtype=dtype, device=device).view(3, 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        bv = np.full(shape=self.bvsize, fill_value=self.bg(), dtype=np.uint8)
        self.__draw_random_lines_on(bv, self.line_counts())
        self.__draw_random_circles_on(bv, self.circ_counts())
        rx, ry, rz = self.rx(), self.ry(), self.rz()
        R = make_rot_mat(rx, ry, rz, rot_order='xyz', dtype=self.dtype, device=self.device)
        H = make_homography(R, self.tz, self.K, bv_pivot=self.bv_pivot, pv_pivot=self.pv_pivot).squeeze().detach().cpu().numpy()
        pv = cv2.warpPerspective(bv, inv(H), self.pvsize)
        return rx, ry, rz, bv, pv

    def __draw_random_lines_on(self, bv, num=1):
        H, W = self.bvsize
        for i in range(num):
            pt1 = (random.randint(0, W), H)
            pt2 = (random.randint(0, W), random.randint(0, H/2))
            cv2.line(bv, pt1, pt2, self.fg(), self.line_width())

    def __draw_random_circles_on(self, view, num=1):
        H, W = self.bvsize
        for i in range(num):
            cxrange = list(range(-W, 0)) + list(range(W, 2*W))
            center = (random.choice(cxrange), random.randint(3*H/4, H))
            radius = random.randint(W/2, H)
            cv2.circle(view, center, radius, self.fg(), self.line_width())

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from bevutils import PerspectiveTransformerLayer

intrinsics = [
    [2304.54786556982,                0, 1686.23787612802], 
    [               0,   2305.875668062, 1354.98496439791], 
    [               0,                0,                1]
]

dtype = torch.float32


img = cv2.imread('sample.jpg')

inp = torch.tensor(img, dtype=dtype)[None, :, :, :]
inps = torch.cat((inp, inp), dim=0).permute(0, 3, 1, 2).cuda()
B, C, H, W = inps.shape

warpPerspective = PerspectiveTransformerLayer((600, 200), (H, W), intrinsics, translate_z=-30, rotation_order='xyz', dtype=dtype)

rx = torch.tensor([1.72, 0.00], requires_grad=True)
rz = torch.tensor([-0.04, 0.00], requires_grad=True)
ry = torch.tensor([0.00, 0.00], requires_grad=False)

bev = warpPerspective(inps, rx, ry, rz).permute(0, 2, 3, 1)
print("requires_grad =", bev.requires_grad)
assert bev.requires_grad == rx.requires_grad | ry.requires_grad | rz.requires_grad

bev = bev.detach().cpu().numpy().astype(np.uint8)

cv2.imshow("sample.py", bev[0])
cv2.waitKey(0)
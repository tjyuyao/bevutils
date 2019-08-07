import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..functional import epipolar as E

class PerspectiveTransformerLayer(nn.Module):

    def __init__(self, bv_size, pv_size, intrinsics, translate_z = 0.0, rotation_order='xyz', device='cuda:0', dtype=torch.float32):
        '''
        `translate_z` is a hyperparameter to be chose in range (-Inf, 1.0), the perspective view will be roughly scaled (1-translate_z) times.
        '''
        super(PerspectiveTransformerLayer, self).__init__()
        self.dtype = dtype
        self.dev = torch.device(device) if device else None
        self.rot_order = rotation_order
        self.bv_size, self.pv_size = bv_size, pv_size
        self.intrinsics = self._prepare_intrinsics(intrinsics)
        self.inv_intrinsics = torch.inverse(self.intrinsics)
        self.bv_pivot, self.pv_pivot = self._prepare_pivots(bv_size, pv_size, self.inv_intrinsics)
        self.n = torch.tensor([[0], [0], [1]], device=self.dev, dtype=self.dtype)
        self.tz = torch.tensor([translate_z], device=self.dev, dtype=self.dtype)
        self.bv_grid = self._prepare_coord_grid(*bv_size)

    def _prepare_intrinsics(self, intrinsics):
        if isinstance(intrinsics, list) or isinstance(intrinsics, np.array):
            intrinsics = torch.tensor(intrinsics, requires_grad=False, device=self.dev, dtype=self.dtype)
        assert isinstance(intrinsics, torch.Tensor)
        assert intrinsics.shape == (3, 3)
        return intrinsics
    
    def _prepare_pivots(self, bv_size, pv_size, inv_intrinsics):
        bv_pivot = torch.tensor([[bv_size[1]/2.0], [bv_size[0]], [1.0]], device=self.dev, dtype=self.dtype)
        pv_pivot = torch.tensor([[pv_size[1]/2.0], [pv_size[0]], [1.0]], device=self.dev, dtype=self.dtype)
        bv_pivot = inv_intrinsics @ bv_pivot
        pv_pivot = inv_intrinsics @ pv_pivot
        return bv_pivot, pv_pivot

    def _prepare_coord_grid(self, H, W):
        xgrid = torch.arange(W, requires_grad=False, device=self.dev, dtype=self.dtype).repeat(H, 1).view((H, W, 1, 1))
        ygrid = torch.arange(H, requires_grad=False, device=self.dev, dtype=self.dtype).unsqueeze_(1).repeat(1, W).view(H, W, 1, 1)
        grid = torch.cat((xgrid, ygrid, torch.ones_like(xgrid, device=self.dev, dtype=self.dtype)), dim=-2)
        return grid

    def forward(self, pv, rx=0.0, ry=0.0, rz=0.0):
        '''
        REFERENCES:
        - Homography: refers to https://en.wikipedia.org/wiki/Homography_(computer_vision)
        - Bilinear Interpolation: refers to https://medium.com/@shanlins/spatial-transformer-networks-stn-and-its-implementation-2638d58d41f8
        '''
        B, C, Hp, Wp, Hb, Wb = *pv.shape, *self.bv_size
        assert B == rx.shape[0]
        # get constrained homography
        R = E.torch.make_rotation_matrix(rx, ry, rz, self.rot_order, device=self.dev, dtype=self.dtype)
        H = E.torch.make_constrained_homography(R, self.tz, self.intrinsics, self.inv_intrinsics, self.bv_pivot, self.pv_pivot)
        # get coordinates on perspective view for each grid: `pv_coord` with shape (B, Hb, Wb, 2, 1)
        bv_grid = self.bv_grid.expand(B, Hb, Wb, 3, 1)
        pv_coord = torch.matmul(H[:, None, None, :, :], bv_grid)
        pv_coord[:, :, :, 0:2, :] /= pv_coord[:, :, :, 2:3, :]
        # gather pixels acoording to `pv_coord`
        x = pv_coord[:,None,:,:,0,0] # with shape (B, 1, Hb, Wb)
        y = pv_coord[:,None,:,:,1,0]
        x0 = x.to(torch.long).clamp_(0, Wp-2)
        y0 = y.to(torch.long).clamp_(0, Hp-2)
        offset_00 = y0 * Wp + x0
        offset_01 = offset_00 + 1
        offset_10 = offset_00 + Wp
        offset_11 = offset_10 + 1
        pv = pv.view(B, C, Hp*Wp) # with shape (B, C, Hp*Wp)
        pvmap = [
            torch.gather(pv, -1, offset_00.expand(B, C, Hb, Wb).view(B, C, Hb*Wb)),
            torch.gather(pv, -1, offset_01.expand(B, C, Hb, Wb).view(B, C, Hb*Wb)),
            torch.gather(pv, -1, offset_10.expand(B, C, Hb, Wb).view(B, C, Hb*Wb)),
            torch.gather(pv, -1, offset_11.expand(B, C, Hb, Wb).view(B, C, Hb*Wb))] # pv maps: with shape (B, C, Hb*Wb)
        # combine pv pixels
        x0, x1, y0, y1 = (x - x0.to(self.dtype)), ((x0+1).to(self.dtype) - x), (y - y0.to(self.dtype)), ((y0+1).to(self.dtype) - y)
        weights = [(x0 * y0), (x0 * y1), (x1 * y0), (x1 * y1)] # weight : with shape (B, 1, Hb, Wb)
        bvmap = sum([w.expand(B, C, Hb, Wb) * p.view(B, C, Hb, Wb) for w, p in zip(weights, pvmap)]) # bvmap with shape (B, C, Hb, Wb)
        mask = (~((x >= 0) & (x < Wp) & (y >= 0) & (y < Hp))).expand(B, C, Hb, Wb)
        bvmap[mask] = 0.0
        return bvmap
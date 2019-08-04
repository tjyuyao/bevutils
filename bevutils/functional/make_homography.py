import torch
from .make_rot_mat import make_rot_mat


def make_homography(R, t, K, K_inv=None, bv_pivot=None, pv_pivot=None):
    assert isinstance(R, torch.Tensor) and R.shape[-2:] == (3, 3)
    assert isinstance(K, torch.Tensor) and K.shape[-2:] == (3, 3)
    B, device, dtype = R.shape[0], R.device, R.dtype
    n = torch.bmm(torch.inverse(R), torch.tensor([[[0], [0], [1]]], dtype=dtype, device=device).expand(B, 3, 1))
    if K_inv is None:
        K_inv = torch.inverse(K)
    if bv_pivot is not None and pv_pivot is not None and t.shape == torch.Size([1]):
        b, p, t2 = K_inv @ bv_pivot.view(3, 1), K_inv @ pv_pivot.view(3, 1), t.expand(B)
        t0 = (R[:,0,2] - b[0]*R[:,2,2] + p[0]*R[:,0,0] + p[1]*R[:,0,1] + b[0]*n[:,2,0]*t2 - b[0]*p[0]*R[:,2,0] - b[0]*p[1]*R[:,2,1] + b[0]*n[:,0,0]*p[0]*t2 + b[0]*n[:,1,0]*p[1]*t2)/(n[:,2,0] + n[:,0,0]*p[0] + n[:,1,0]*p[1])
        t1 = (R[:,1,2] - b[1]*R[:,2,2] + p[0]*R[:,1,0] + p[1]*R[:,1,1] + b[1]*n[:,2,0]*t2 - b[1]*p[0]*R[:,2,0] - b[1]*p[1]*R[:,2,1] + b[1]*n[:,0,0]*p[0]*t2 + b[1]*n[:,1,0]*p[1]*t2)/(n[:,2,0] + n[:,0,0]*p[0] + n[:,1,0]*p[1])
        t = torch.cat((t0[:, None, None], t1[:, None, None], t2[:, None, None]), dim=1)
    elif t.shape == torch.Size([3, 1]) or t.shape == torch.Size([3, 1]):
        t = t.view(1, 3, 1).expand(B, 3, 1)
    elif t.shape == torch.Size([B, 3, 1]) or t.shape == torch.Size([B, 3]):
        t = t.view(B, 3, 1)
    else:
        raise NotImplementedError("unsupported shape of t")
    H = K @ (R - t @ n.transpose(1, 2)) @ K_inv
    return H
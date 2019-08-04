import torch


def make_rot_mat(rx, ry, rz, rot_order='xyz', dtype=None, device=None):
    if ~isinstance(rx, torch.Tensor):
        rx = torch.tensor([rx], dtype=dtype, device=device)
        ry = torch.tensor([ry], dtype=dtype, device=device)
        rz = torch.tensor([rz], dtype=dtype, device=device)
    cx, cy, cz = torch.cos(rx), torch.cos(ry), torch.cos(rz)
    sx, sy, sz = torch.sin(rx), torch.sin(ry), torch.sin(rz)
    B = rx.shape[0]
    _Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(B, 1, 1)
    _Rx[:, 1, 1], _Rx[:, 1, 2], _Rx[:, 2, 1], _Rx[:, 2, 2] = cx, -sx, sx, cx
    _Ry = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(B, 1, 1)
    _Ry[:, 0, 0], _Ry[:, 0, 2], _Ry[:, 2, 0], _Ry[:, 2, 2] = cy, -sy, sy, cy
    _Rz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(B, 1, 1)
    _Rz[:, 0, 0], _Rz[:, 0, 1], _Rz[:, 1, 0], _Rz[:, 1, 1] = cz, -sz, sz, cz
    _R = {'x' : _Rx, 'y' : _Ry, 'z' : _Rz}
    R = torch.eye(3, device=device, dtype=dtype).repeat(B, 1, 1)
    for i in rot_order:
        R = torch.bmm(R, _R[i])
    return R
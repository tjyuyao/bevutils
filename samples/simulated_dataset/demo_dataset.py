import cv2
from bevutils.datasets import SimulatedDataSet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

dataset_kwargs = {
    "length":2,
    "bv_size":(1000,500),
    "pv_size":(480, 640),
    "translate_z":-10.0,
}

dataset = SimulatedDataSet(**dataset_kwargs)

plt.figure(figsize=(20,10))

for batch_ndx, sample in enumerate(dataset):
    (pv, rx0, ry0, rz0), (bv, rx, ry, rz) = sample
    H0 = dataset.make_constrained_homography(rx0, ry0, rz0)
    bv0 = cv2.warpPerspective(pv, H0, (500, 1000))
    H1 = dataset.make_constrained_homography(rx, ry, rz)
    bv1 = cv2.warpPerspective(pv, H1, (500, 1000))
    plt.subplot(2, 4, 4*batch_ndx+1)
    plt.imshow(bv, cmap='gray')
    plt.subplot(2, 4, 4*batch_ndx+2)
    plt.imshow(pv, cmap='gray')
    plt.subplot(2, 4, 4*batch_ndx+3)
    plt.imshow(bv0, cmap='gray')
    plt.subplot(2, 4, 4*batch_ndx+4)
    plt.imshow(bv1, cmap='gray')
    plt.title("rx=%f\nry=%f\nrz=%f" % (rx, ry, rz))

plt.savefig("tmp.png")
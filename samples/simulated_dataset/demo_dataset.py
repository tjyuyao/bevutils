from bevutils.datasets import SimulatedDataSet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

dataset = SimulatedDataSet(length=5)

loader = DataLoader(dataset, batch_size=1)

plt.figure(figsize=(20,10))

for batch_ndx, sample in enumerate(loader):
    pv, (bv, rx, ry, rz) = sample
    plt.subplot(2, 5, batch_ndx+1)
    plt.imshow(bv[0], cmap='gray')
    plt.subplot(2, 5, 5+batch_ndx+1)
    plt.imshow(pv[0], cmap='gray')
    plt.title("rx=%f\nry=%f\nrz=%f" % (rx, ry, rz))

plt.show()
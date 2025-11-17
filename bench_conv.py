import torch
import torch.nn as nn
import time
import timeit
from statistics import median

device = torch.device("cuda")

model = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU()
).to(device)

# Warm up
x = torch.randn(128, 64, 128, 128, dtype=torch.float32).to(device)
y = model(x)
torch.cuda.synchronize()

def test(model,x,torch):
    def run():
        y = model(x)
        torch.cuda.synchronize()
    return run

sizes = [2 ** i for i in range(5,9)] # 32 - 256
time_median = [0] * len(sizes)

fname = "pytorch-" + torch.cuda.get_device_name().replace(" ","-") + ".csv"
f = open(fname,"w")
print("size","median_time_second",file=f,sep=",")

for i in range(len(sizes)):
    x = torch.randn(128, 64, sizes[i], sizes[i], dtype=torch.float32).to(device)
    time_median[i] = median(timeit.repeat(test(model,x,torch),number=1,repeat = 1000))
    print("size ",sizes[i]," time ",time_median[i] * 1e3," ms")
    print(sizes[i],time_median[i],file=f,sep=",")

f.close()

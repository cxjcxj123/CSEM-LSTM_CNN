import torch
from torch.autograd import Variable as V
import importlib
import numpy as np
from tqdm import tqdm

SOURCE_PATH = r".\Field_data.npy"
TARGET_PATH = r'.\Unet_LSTM_Result.txt'
MODEL_FILENAME = r'.\Fitted_Model.pt'
MODEL_NAME = r'Unet_LSTM'


def norm_csamt(data):
    return (data - 1.8661 + 1e-8) / ((5 - 1.8661 + 1e-8))


def one_image(net, data):
    data = np.abs(data)
    origindata = np.log10(data)
    origindata = norm_csamt(origindata)

    origindata = np.expand_dims(origindata, axis=1)
    origindata = np.expand_dims(origindata, axis=2)
    img = origindata.transpose(1, 2, 0)
    img = img * 1.0

    with torch.no_grad():
        img = V(torch.Tensor(img).cuda())
        mask = net.forward(img).squeeze().cpu().data.numpy()
    return mask


csamt_file = np.load(SOURCE_PATH)
image_list = list(np.linspace(0, 447, 447))
net = getattr(importlib.import_module('network.unet_lstm'), MODEL_NAME)
net = net().cuda()
net.load_state_dict(torch.load(MODEL_FILENAME))
net.eval()
pbar = tqdm(total=len(image_list))  # 进度条
for i, name in enumerate(image_list):
    mask = one_image(net, csamt_file[i])
    mask = (mask * ((4.572 - (1.8215 + 1e-8)))) + (1.8215 + 1e-8)
    mask = 10 ** mask
    with open(TARGET_PATH, 'a') as file:
        file.write(' '.join(map(str, mask)) + '\n')
    pbar.update(1)
pbar.close()

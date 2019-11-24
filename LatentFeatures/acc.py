import torch

PATH = '../checkpoints/LFNet.pth'

infos = torch.load(PATH, map_location='cpu')
print(infos['ACCURACY'])

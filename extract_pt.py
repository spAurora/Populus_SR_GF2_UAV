import torch

s = torch.load(r'D:\github\Populus_SR_GF2_UAV\results\20250714-231224-gupopulus_x8\checkpoints/checkpoint.pt', map_location=torch.device('cpu'),weights_only=True)
t = {}
for k, v in s['model'].items():
    if k.startswith('model.'):
        t[k[6:]] = v
torch.save(t, 'pretrained-rrdbnet-gupopulus-x8.pt')
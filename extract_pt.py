import torch

s = torch.load(r'D:\github\Populus_SR_GF2_UAV\results\20250328-204222-parcel_s2_250328/checkpoints/checkpoint.pt', map_location=torch.device('cpu'))
t = {}
for k, v in s['model'].items():
    if k.startswith('model.'):
        t[k[6:]] = v
torch.save(t, 'pretrained-rrdbnet-parcel_s2.pt')
import torch

s = torch.load(r'D:\github\Populus_SR_GF2_UAV\results\20250329-143157-parcel_gf2_250329/checkpoints/checkpoint.pt', map_location=torch.device('cpu'))
t = {}
for k, v in s['model'].items():
    if k.startswith('model.'):
        t[k[6:]] = v
torch.save(t, 'pretrained-rrdbnet-parcel_gf2.pt')
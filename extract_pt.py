import torch

s = torch.load('results/20241227-210034-gupopulus_241227/checkpoints/checkpoint.pt', map_location=torch.device('cpu'))
t = {}
for k, v in s['model'].items():
    if k.startswith('model.'):
        t[k[6:]] = v
torch.save(t, 'pretrained-rrdbnet-gupopulus.pt')
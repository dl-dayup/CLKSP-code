import torch
tes=torch.load('tes.pt').unsqueeze(0)
exa=torch.load('exa.pt').unsqueeze(0)
dru=torch.load('dru.pt').unsqueeze(0)
sit=torch.load('sit.pt').unsqueeze(0)
sur=torch.load('sur.pt').unsqueeze(0)
dis=torch.load('dis.pt').unsqueeze(0)
prom=torch.concat([tes,exa,dru,sit,sur,dis],0)
print(prom.shape)

a=['a','b','c']
b=[1,2,3]
for index,i in enumerate(b):
    print(index,i)
for i in a:

    b.append(i)
for index,i in enumerate(b):
    print(index,i)


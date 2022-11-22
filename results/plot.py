import matplotlib.pyplot as plt
import numpy as np

with open('prune.txt','r')as f:
    prune_model = f.readlines()
    prune_model = [i.replace('\n','') for i in prune_model]
    prune_model = np.array(prune_model,dtype=float)

with open('m.txt','r')as f:
    mix_model = f.readlines()
    mix_model = [i.replace('\n','') for i in mix_model]
    mix_model = np.array(mix_model,dtype=float)

with open('b.txt','r')as f:
    bias_model = f.readlines()
    bias_model = [i.replace('\n','') for i in bias_model]
    bias_model = np.array(bias_model,dtype=float)

with open('l.txt','r')as f:
    label_model = f.readlines()
    label_model = [i.replace('\n','') for i in label_model]
    label_model = np.array(label_model,dtype=float)

with open('p.txt','r')as f:
    group_model = f.readlines()
    group_model = [i.replace('\n','') for i in group_model]
    group_model = np.array(group_model,dtype=float)

with open('n.txt','r')as f:
    new_model = f.readlines()
    new_model = [i.replace('\n','') for i in new_model]
    new_model = np.array(new_model,dtype=float)

with open('resnet.txt','r')as f:
    s = f.readlines()
    s = [i.replace('\n','') for i in s]
    s = np.array(s,dtype=float)
    
plt.figure()
plt.plot(prune_model,label='50% pruning')
# plt.plot(mix_model,label='mix')
# plt.plot(bias_model,label='bias')
# plt.plot(label_model,label='label smoothing')
# plt.plot(group_model,label='group convolution')
# plt.plot(new_model,label='new')
plt.plot(s,label='resnet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

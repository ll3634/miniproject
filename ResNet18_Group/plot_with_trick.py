import matplotlib.pyplot as plt
import numpy as np

with open('group_after_trick.txt','r')as f:
    group_model = f.readlines()
    group_model = [i.replace('\n','') for i in group_model]
    group_model = np.array(group_model,dtype=float)


with open('resnet.txt','r')as f:
    s = f.readlines()
    s = [i.replace('\n','') for i in s]
    s = np.array(s,dtype=float)
    
plt.figure()
plt.plot(group_model,label='Grouped Conv')
plt.plot(s,label='ResNet18')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

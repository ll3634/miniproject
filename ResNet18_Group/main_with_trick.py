'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


from torchsummary import summary
from label_smoothing import LSR

#--------------------------------------------------------------------------change1 start
import torch.utils.data as data
import copy
#--------------------------------------------------------------------------change1 finish

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best valid(test) accuracy -----------------------------------change
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#--------------------------------------------------------------------------change2 start
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

#add
valid_portion = 0.9
number_train = int(len(trainset) * valid_portion)
number_valid = len(trainset) - number_train
trainset, validset = data.random_split(trainset,
    [number_train, number_valid])

validset = copy.deepcopy(validset)
validset.dataset.transform = transform_test

#batch_size
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

#shuffle?
validloader = torch.utils.data.DataLoader(
    validset, batch_size=100, shuffle=False, num_workers=2)
#--------------------------------------------------------------------------change2 finish


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18_group()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_group_after_trick.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# criterion = nn.CrossEntropyLoss()
criterion = LSR()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # inputs, targets = inputs.to(device), targets.to(device)
        
        inputs, targets_a, targets_b, lam = mixup_data(inputs.to(device), targets.to(device), args.alpha, torch.cuda.is_available())

        optimizer.zero_grad()
        
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)

        outputs = net(inputs)
        
        # loss = criterion(outputs, targets)
        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss/(batch_idx+1)
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

#--------------------------------------------------------------------------change3 start
def valid(epoch):
    global best_acc
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    
    with open('group_after_trick.txt','a+')as f:
        f.write('%.5f'%acc)
        f.write('\n')
    print('Epoch:',epoch,'acc:', acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_group_after_trick.pth')
        best_acc = acc
    return acc
    
#     #plot1
#     with open('acc_group.txt','a+')as f:
#         f.write('%.5f'%acc)
#         f.write('\n')
#     print('Epoch:',epoch,'acc:', acc)
    
#     if acc > best_acc:
#         print('(Validation) Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
# #----------------------------------------------------------------------add start
#         torch.save(net.state_dict(),'net_saved')
# #----------------------------------------------------------------------add finish
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt_group.pth')
#         best_acc = acc
        
#         #plot3
#         #return acc


def test(epoch):
    print('\nTest epoch: %d' % epoch)
    print('Test Acc:')
#----------------------------------------------------------------------add start
    # net.load_state_dict(torch.load('net_saved'))
    net.load_state_dict(torch.load('./checkpoint/ckpt_group_after_trick.pth')['net'])
#----------------------------------------------------------------------add finish
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100.*correct/total
#    with open('test_group_after_trick.txt','a+')as f:
#        f.write('%.5f'%acc)
#        f.write('\n')
    print('Epoch:',epoch,'acc:', acc)

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            
            
#--------------------------------------------------------------------------change3 finish
print(summary(net, input_size = (3, 32, 32)))

    
for epoch in range(start_epoch, start_epoch+178):
    train(epoch)
#---------------------------------------------------------------------------------------change4 start
    valid(epoch)
#--------------------------------------------------------------------------------------change4 finish
    scheduler.step()

# ----------------------------------------------------------------------------------------change5 start

    
for epoch in range(start_epoch, start_epoch+2):
    test(epoch)
    
#------------------------------------------------------------------------------------------change5 finish

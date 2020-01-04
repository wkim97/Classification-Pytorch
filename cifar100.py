import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#############################################################################
# 1. Get training and test data sets
#############################################################################
transform = transforms.Compose(
    [transforms.ToTensor()])
batch_size = 50
trainset = torchvision.datasets.CIFAR100(
    './data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR100(
    './data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
classes = ['apple',
           'aquarium_fish',
           'baby',
           'bear',
           'beaver',
           'bed',
           'bee',
           'beetle',
           'bicycle',
           'bottle',
           'bowl',
           'boy',
           'bridge',
           'bus',
           'butterfly',
           'camel',
           'can',
           'castle',
           'caterpillar',
           'cattle',
           'chair',
           'chimpanzee',
           'clock',
           'cloud',
           'cockroach',
           'couch',
           'crab',
           'crocodile',
           'cup',
           'dinosaur',
           'dolphin',
           'elephant',
           'flatfish',
           'forest',
           'fox',
           'girl',
           'hamster',
           'house',
           'kangaroo',
           'computer_keyboard',
           'lamp',
           'lawn_mower',
           'leopard',
           'lion',
           'lizard',
           'lobster',
           'man',
           'maple_tree',
           'motorcycle',
           'mountain',
           'mouse',
           'mushroom',
           'oak_tree',
           'orange',
           'orchid',
           'otter',
           'palm_tree',
           'pear',
           'pickup_truck',
           'pine_tree',
           'plain',
           'plate',
           'poppy',
           'porcupine',
           'possum',
           'rabbit',
           'raccoon',
           'ray',
           'road',
           'rocket',
           'rose',
           'sea',
           'seal',
           'shark',
           'shrew',
           'skunk',
           'skyscraper',
           'snail',
           'snake',
           'spider',
           'squirrel',
           'streetcar',
           'sunflower',
           'sweet_pepper',
           'table',
           'tank',
           'telephone',
           'television',
           'tiger',
           'tractor',
           'train',
           'trout',
           'tulip',
           'turtle',
           'wardrobe',
           'whale',
           'willow_tree',
           'wolf',
           'woman',
           'worm']

mapping = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

mapping_values = mapping.values()
combined_mapping = []
for lists in mapping_values:
    combined_mapping.extend(lists)
mapping_keys = ['aquatic mammals',
                'fish',
                'flowers',
                'food containers',
                'fruit and vegetables',
                'household electrical device',
                'household furniture',
                'insects',
                'large carnivores',
                'large man-made outdoor things',
                'large natural outdoor scenes',
                'large omnivores and herbivores',
                'medium-sized mammals',
                'non-insect invertebrates',
                'people',
                'reptiles',
                'small mammals',
                'trees',
                'vehicles 1',
                'vehicles 2']

#############################################################################
# 1-1. Show randomly selected training images and labels
#############################################################################
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
# imshow(torchvision.utils.make_grid(images))

#############################################################################
# 2. Define CNN, loss, and optimizer
#############################################################################
class CIFAR100_net(nn.Module):
    def __init__(self):
        super(CIFAR100_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 100)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 32 * 5 * 5)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


net = CIFAR100_net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#############################################################################
# 3. Learn the model with training data
#############################################################################
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i, running_loss / 100))
            running_loss = 0.0
print('Finished Training')
PATH = './CIFAR100_net.pth'
torch.save(net.state_dict(), PATH)

#############################################################################
# 4. Testing with the test data
#############################################################################
test_net = CIFAR100_net()
test_net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = test_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
print('Accuracy of the network on the test images: %d %%'
      % (100 * correct / total))
for i in range(100):
    print('Accuracy of %s-\t%s:\t%2d %%'
          % (mapping_keys[int(combined_mapping.index(classes[i])/5)],
             classes[i], 100 * class_correct[i] / class_total[i]))

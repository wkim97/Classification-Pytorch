import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stanford_dogs_data import dogs
import configs

#############################################################################
# 1. Get training and test data sets
#############################################################################
transform = transforms.Compose([
            transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
batch_size = 100
trainset = dogs(root=configs.imagesets,
                train=True, cropped=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = dogs(root=configs.imagesets,
                train=False, cropped=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
classes = trainset.classes

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
class dog_net(nn.Module):
    def __init__(self):
        super(dog_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.conv3 = nn.Conv2d(64, 256, 4)
        self.fc1 = nn.Linear(256 * 25 * 25, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 120)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 256 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
net = dog_net()

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#############################################################################
# 3. Learn the model with training data
#############################################################################
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        labels = labels.type(torch.LongTensor)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i, running_loss / 1000))
            running_loss = 0.0
print('Finished Training')
PATH = './stanford_dogs.pth'
torch.save(net.state_dict(), PATH)

#############################################################################
# 4. Testing with the test data
#############################################################################
test_net = dog_net()
test_net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
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
for i in range(10):
    print('Accuracy of %s %2d %%'
          % (classes[i], 100 * class_correct[i] / class_total[i]))
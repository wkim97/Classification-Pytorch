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
trainset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

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
class MNIST_net(nn.Module):
    def __init__(self):
        super(MNIST_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x
net = MNIST_net()

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
                  (epoch + 1, i, running_loss / 1000))
            running_loss = 0.0
print('Finished Training')
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH)

#############################################################################
# 4. Testing with the test data
#############################################################################
test_net = MNIST_net()
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
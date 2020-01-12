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
batch_size = 1
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

# #############################################################################
# # 3. Learn the model with training data
# #############################################################################
# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         images, labels = data
#         optimizer.zero_grad()
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 100 == 0:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i, running_loss / 1000))
#             running_loss = 0.0
# print('Finished Training')
PATH = './MNIST_net.pth'
# torch.save(net.state_dict(), PATH)

#############################################################################
# 4. Fast Gradient Sign Attack
#############################################################################
epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

# image = original clean image x
# epsilon = pixel-wise perturbation amount
# data_grad = gradient of the loss w.r.t. input image
# perturbed_image = image + epsilon * data_grad
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

#############################################################################
# 5. Testing with the test data
#############################################################################
test_net = MNIST_net()
test_net.load_state_dict(torch.load(PATH))
accuracies = []
adversarial_images = []
for eps in epsilons:
    correct = 0
    total = 0
    adv_examples = []
    for data in testloader:
        total += 1
        original_image, labels = data
        original_image.requires_grad = True
        original_outputs = test_net(original_image)
        _, init_pred = torch.max(original_outputs.data, 1)
        loss = criterion(original_outputs, labels)
        test_net.zero_grad()
        loss.backward()
        # perturbed image
        data_grad = original_image.grad.data
        perturbed_data = fgsm_attack(original_image, eps, data_grad)
        perturbed_outputs = test_net(perturbed_data)
        _, final_pred = torch.max(perturbed_outputs, 1)
        if init_pred == labels and final_pred == init_pred:
            correct += 1
        adversarial_image = perturbed_data.squeeze().detach().numpy()
        if total == 1:
            adversarial_images.append(adversarial_image)
    eps_accuracy = correct / float(total)
    print("Eps:{}\tAccuracy:{}".format(eps, eps_accuracy))
    accuracies.append(eps_accuracy)

plt.title("Epsilon vs Accuracy")
plt.plot(epsilons, accuracies, "rs-")
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.show()

fig = plt.figure()
rows = 1
cols = len(adversarial_images)
cnt = 1
for i in range(len(adversarial_images)):
    ax = fig.add_subplot(rows, cols, cnt)
    ax.imshow(adversarial_images[i], cmap="gray")
    ax.set_xlabel("{}".format(epsilons[i]))
    ax.set_xticks([])
    ax.set_yticks([])
    cnt += 1
plt.show()
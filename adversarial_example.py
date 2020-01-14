import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#############################################################################
# 0. Fast Gradient Sign Attack
#############################################################################
# epsilons = [0]
# epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
# epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# epsilons = [100, 150, 200, 250, 300, 350, 400]
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# image = original clean image x
# epsilon = pixel-wise perturbation amount
# data_grad = gradient of the loss w.r.t. input image
# perturbed_image = image + epsilon * data_grad
def fgsm_attack(image, label, epsilon, model):
    outputs = model(image)
    loss = criterion(outputs, label)
    loss.backward()
    data_grad = image.grad.data
    sign_data_grad = data_grad.sign()
    p_image = image + epsilon * sign_data_grad
    p_image = torch.clamp(p_image, 0, 1)
    return p_image

def iter_fgsm_attack(image, label, epsilon, n_iter, model):
    orig = image
    for i in range(n_iter):
        print(i)
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        data_grad = image.grad.data
        sign_grad_data = data_grad.sign()
        perturbation = epsilon * sign_grad_data
        # perturbation = torch.clamp((image.data + perturbation) - orig, min=-epsilon/255.0, max=epsilon/255.0)
        p_image = orig + perturbation
        p_image = torch.clamp(p_image, 0, 1)
        image = p_image
        data_grad.zero_()
    return image

#############################################################################
# 1. Get training and test data sets
#############################################################################
transform = transforms.Compose(
    [transforms.ToTensor()])
train_batch_size = 100
test_batch_size = 100
trainset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=train_batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=True)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

#############################################################################
# 1-1. Show randomly selected training images and labels
#############################################################################
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

# Model using RBF from https://github.com/csnstat/rbfn
class RbfNet(nn.Module):
    def __init__(self, centers, num_class=10):
        super(RbfNet, self).__init__()
        self.centers = centers
        self.num_centers = centers.size(0)
        self.num_class = num_class

        self.linear = torch.nn.Linear(self.num_centers, self.num_class, bias=True)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers) / 10)

    def radial_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False).sqrt()))
        return C

    def forward(self, batches):
        radial_val = self.radial_fun(batches)
        class_score = self.linear(radial_val)
        return class_score

# batch_images, batch_labels = next(iter(trainloader))
# centers = batch_images
# net = RbfNet(centers, num_class=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#############################################################################
# 3. Learn the model with training data
#############################################################################
for eps in epsilons:
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images.requires_grad = True
            images = fgsm_attack(images, labels, eps, net)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    #         if i % 100 == 0:
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i, running_loss / 1000))
    #             running_loss = 0.0
    # print('Finished Training')
    PATH = './MNIST_net.pth'
    # PATH = './MNIST_net_rbf.pth'
    torch.save(net.state_dict(), PATH)

    #############################################################################
    # 5. Testing with the test data
    #############################################################################
    test_net = MNIST_net()

    # batch_images, batch_labels = next(iter(testloader))
    # centers = batch_images
    # test_net = RbfNet(centers, num_class=10)

    test_net.load_state_dict(torch.load(PATH))
    accuracies = []
    error_rate = []
    adversarial_images = []
    for test_eps in epsilons:
        correct = 0
        error = 0
        total = 0
        error_total = 0
        adv_examples = []
        for data in testloader:
            total += test_batch_size
            original_image, labels = data
            original_image.requires_grad = True
            original_outputs = test_net(original_image)
            _, init_pred = torch.max(original_outputs, 1)

            # For iter_fgsm
            perturbed_image, labels = data
            perturbed_image.requires_grad = True
            perturbed_outputs = test_net(perturbed_image)

            # perturbed image
            perturbed_data = fgsm_attack(original_image, labels, test_eps, test_net)
            n_iter = 100
            # perturbed_data = iter_fgsm_attack(perturbed_image, labels, eps, n_iter, test_net)
            perturbed_outputs = test_net(perturbed_data)
            _, final_pred = torch.max(perturbed_outputs, 1)

            init_correct = (init_pred == labels)
            adv_success = (final_pred != init_pred)
            adv_failure = (final_pred == init_pred)
            error_total += init_correct.sum()
            correct += (init_correct * adv_failure).sum()
            error += (init_correct * adv_success).sum()
            adversarial_image = perturbed_data.squeeze().detach().numpy()
            if total == 1:
                adversarial_images.append(adversarial_image)

        eps_accuracy = correct.numpy() / float(total)
        eps_error_rate = error.numpy() / float(error_total.numpy())
        print("Trained with eps:{}\tTested with eps:{}\tAccuracy:{}\tError rate:{}".format(eps, test_eps, eps_accuracy, eps_error_rate))
        accuracies.append(eps_accuracy)
        error_rate.append(eps_error_rate)

plt.title("Epsilon vs Accuracy")
plt.plot(epsilons, accuracies, "rs-")
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.show()

plt.title("Epsilon vs Error rate")
plt.plot(epsilons, error_rate, "rs-")
plt.xlabel('Epsilon')
plt.ylabel('Error rate')
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
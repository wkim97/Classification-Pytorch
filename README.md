# Classification-Pytorch

Various image classification using Pytorch done as an exercise

To configure interpreter at Pycharm to use Pytorch packages:
1. **Ctrl + Alt + S**
2. Select **Project <Project Name>**, and then **Project Interpreter**.
3. Click on **Existing Project** and select the appropriate virtual environment where pytorch is installed.  
  (Current directory: D:\Anaconda3\envs\pytorch_env\python.exe)

***

### Part A - MNIST Dataset
Uses MNIST dataset from Pytorch package using Convolutional Neural Network designed by self.

Uses 2 convolutional layers with 2 pooling kernels, 1 fully connected linear layer, and 1 softmax.

Results produced are 98% accuracy on test dataset.

Code provided in ***mnist.py***.

### Part B - Stanford Dogs Dataset
Uses Stanford dogs dataset from http://vision.stanford.edu/aditya86/ImageNetDogs/.

Due to large input image sizes, I designed a CNN model with 3 convolutional layers with 3 pooling kernels, 3 fully connected linear layer, and a softmax layer.

Failed to produce valid results due to limitation of running without GPU.

Code provided in ***stanford_dogs.py***.

### Part C - CIFAR 100 Dataset
Uses CIFAR 100 dataset from Pytorch package using Convolutional Neural Network designed by self.

Designed 6 different CNN models and tested accuracy for each model - gained accuracy of up to ~50% on test dataset.

Faced restriction of running only with CPU - could not implement more complex models. Model that runs with highest accuracy 
is designed with 3 convolutional layers with 3 pooling kernels, 2 fully connected linear layers, and a 
softmax layer.

Code provided in ***cifar100.py***

### Part D - Adversarial Images 
Uses Fast Gradient Sign Method Attack (FGSM) to add adversarial noise to MNIST dataset with CNN model using 2 convolutional layers with 
2 poolings, 1 fully connected linear layer, and a softmax layer.

To create adversarial images, I applied the formula - perturbed_image = original_image + epsilon * sign_of_gradient - to all 
test images and compared the changed label output with original label output.

Results show that FGSM does indeed lower accuracy of labeling of test images. As epsilon value increases, the accuracy decreases, as shown below:
![image](./Accuracy%20vs%20Epsilon%20Epsilons%200-0.06.png)
![image](./Accuracy%20vs%20Epsilon%20Epsilons%200-0.30.png)

However, the change (addition of noise) becomes more apparent to human eyes as epsilon increases, as shown below:
![image](./Classification%20Results%20Epsilons%200-0.06.png)
![image](./Classification%20Results%20Epsilons%200-0.30.png)

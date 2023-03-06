# Pytorch-CIFAR10

Implement the structure-of-CIFAR10-quick-model using pytorch

## Requirement

- torch~=1.10.0+cu102

- torchvision~=0.11.0+cu102

- pillow~=9.2.0

## Usage

1. If your computer has a GPU, run the file **GPU_train.py**，Otherwise, run the file **CPU_train.py**.

2. You can set the epoch in the code yourself to get the generated model.
   
   > I set the epoch value to 30

3. Select the model with the highest test_accuary and load it into the Test.py for testing

Running the code automatically downloads the dataset from the [official website]([CIFAR-10 and CIFAR-100 datasets (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)) to your local computer.

After training and testing, a logs folder is generated. Use the following instructions in the terminal to be able to visualize

```
tensorboard --logdir=logs
```

## Result

The training results of the structure-of-CIFAR10-quick-model are okay, and the model obtained through 30 rounds of training can have a maximum test_accuracy of 65.47%

### Indices corresponding to ten types of objects

![Ten classes](/cifar10/classes.png)

## Introduction

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.  

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:

![10 classes](/cifar10/10%20classes.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## Structure-of-CIFAR10-quick-model

![Structure-of-CIFAR10-quick-model](/cifar10/Structure-of-CIFAR10-quick-model.png)

## Calculation formula

[It is available in the official pytorch documentation]([Conv2d — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d))

This formula is used to calculate the padding value in a convolution operation

![Calculation formula](/cifar10/Calculation%20formula.png)

Calculate the padding value：

![Calculation](/cifar10/Calculation%20.png)

Stride can only be 1, otherwise the value of padding is too large. 

Thus stride=1, padding=2

## The result after model training

With the help of tensorboard

### Train_loss

![train_loss](/cifar10/train_loss.png)

### Test_loss

![test_loss](/cifar10/test_loss.png)

### Test_accuracy

![test_accuracy](/cifar10/test_accuracy.png)

### Network structure

![1](/cifar10/1.png)

![2](/cifar10/2.png)

![3](/cifar10/3.png)

![4](/cifar10/4.png)

![5](/cifar10/5.png)

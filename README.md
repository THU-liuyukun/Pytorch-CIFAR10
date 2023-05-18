# Pytorch-CIFAR10

## 0. 介绍

使用 pytorch 实现 structure-of-CIFAR10-quick-model

## 1. 依赖

- torch~=1.10.0+cu102

- torchvision~=0.11.0+cu102

- pillow~=9.2.0

## 2. 使用

1. 如果你的电脑有 GPU，运行文件GPU_train.py，否则，运行文件CPU_train.py。

2. 你可以设置代码中的epoch来生成的模型。
   
   > 我将epoch的值设为30

3. 选择test_accuary最高的模型加载到Test.py中进行测试。

运行代码会自动从[CIFAR-10 and CIFAR-100 datasets (toronto.edu)](https://www.cs.toronto.edu/~kriz/cifar.html)下载数据集。

经过训练和测试后，目录中会生成一个logs文件夹。在终端中使用以下指令可以可视化：

```
tensorboard --logdir=logs
```

## 3. 数据集介绍

CIFAR-10 数据集由 10 个类别的 60000 张 32x32 彩色图像组成，每个类别有 6000 张图像。有50000张训练图像和10000张测试图像。

数据集分为5个训练批次和1个测试批次，每个批次有 10000 张图像。测试批次恰好包含从每个类别中随机选择的 1000 张图像。训练批次包含随机顺序的剩余图像，但一些训练批次可能包含来自一个类别的图像多于另一个类别。在它们之间，训练批次恰好包含来自每个类别的 5000 张图像。

以下是数据集中的类别，以及每个类别的 10 张随机图像：

![10 classes](/cifar10/10%20classes.png)

这些类是完全互斥的。汽车和卡车之间没有重叠。“汽车”包括轿车、SUVs 之类的东西。“卡车”仅包括大卡车。两者都不包括皮卡车。

### 3.1 十类对象对应的索引

![Ten classes](/cifar10/classes.png)

## 4. Structure-of-CIFAR10-quick-model

![Structure-of-CIFAR10-quick-model](/cifar10/Structure-of-CIFAR10-quick-model.png)

## 5. 计算公式

[Conv2d — PyTorch 1.13 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

This formula is used to calculate the padding value in a convolution operation
此公式可用于计算卷积运算中的padding

![Calculation formula](/cifar10/Calculation%20formula.png)

计算padding：

![Calculation](/cifar10/Calculation%20.png)

stride只能为1，否则padding的值太大。

因此stride=1，padding=2

## 6. 结果

structure-of-CIFAR10-quick-model的训练结果还可以，经过30轮训练得到的模型最大test_accuracy可以达到65.47%

### 6.1 使用tensorboard可视化

#### 6.1.1 Train_loss

![train_loss](/cifar10/train_loss.png)

#### 6.1.2 Test_loss

![test_loss](/cifar10/test_loss.png)

#### 6.1.3 Test_accuracy

![test_accuracy](/cifar10/test_accuracy.png)

#### 6.1.4 Network structure

![1](/cifar10/1.png)

![2](/cifar10/2.png)

![3](/cifar10/3.png)

![4](/cifar10/4.png)

![5](/cifar10/5.png)

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter

img_path = "./images/frog.png"
img = Image.open(img_path)
# 将png的4通道变为3通道
img = img.convert("RGB")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

img = transform(img)
print(img.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

img = torch.reshape(img, (1, 3, 32, 32))

# 用训练好的模型测试，通过tensorboard观察到准确率最高的模型是gpu_train_model29.pth
# 1、将GPU训练的模型，加载到GPU上
classifier = torch.load("./models/gpu_train_model29.pth")
img = img.to(device)

# 2、将GPU上训练的模型，加载到CPU上，需要map_location指定
# model = torch.load("./models/gpu_train_model29.pth", map_location=torch.device("cpu"))
# 用CPU来测试，不需要img = img.to(device)

writer = SummaryWriter("logs")

classifier.eval()
with torch.no_grad():
    output = classifier(img)
    writer.add_graph(classifier, img)

print(output)
print(output.argmax(1))
writer.close()
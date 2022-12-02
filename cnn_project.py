import torch
from torchvision.transforms import transforms
import torchvision
import torch.nn as nn
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, Lambda
from skimage import io, transform
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Fetal_Data_Set(Dataset):
    def __init__(self,annotation_file,file_path,transform=None,target_transform=None):
        self.img_labels=pd.read_excel(annotation_file)
        self.file_path=file_path
        self.transform=transform
        self.target_transform=target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.file_path,self.img_labels.iloc[idx,0])
        loader=transforms.Compose([transforms.ToTensor()])
        image=Image.open(img_path).convert('L')
        image=loader(image)
        image=image.to(torch.float)
        

        label=self.img_labels.iloc[idx,2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_dataset=Fetal_Data_Set("F:\\fetal plane\\train_label.xlsx", 'F:\\fetal plane\\Train',transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((255,255)),
    transforms.ToTensor()]))
test_dataset=Fetal_Data_Set("F:\\fetal plane\\test_label.xlsx", 'F:\\fetal plane\\Test',transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((255,255)),
    transforms.ToTensor()]))


train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))


img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


num_epochs=100
num_classes = 6
learning_rate = 0.001


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),# the kernel/filter
            nn.Sigmoid(),
            nn.BatchNorm2d(16),
            nn.ReLU(),#activation function
            nn.MaxPool2d(kernel_size=2, stride=2))#pooling layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes) #out layer
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
#forward propogation
#this is a block including nerual network and forward propogation
#if we want define a block, first we have to define the nerual network in construction function,
# then we forward the neural network.\
 


model = ConvNet(num_classes).to(device)
#realize the block and send to the device ready for procecss


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#first we have to determine the length of our training sample
#

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
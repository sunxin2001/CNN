import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class My_dataset(Dataset):
    def __init__(self,file_path,transform=None,target_transform=None):
        self.file_path=file_path
        self.transform=transform
        self.target_transform=target_transform
        self.images=os.listdir(file_path)
        
    def __getitem__(self,index):
        image_index=self.images[index]
        img_path=os.path.join(self.file_path,image_index)
        loader=transforms.Compose([transforms.ToTensor()])
        image=Image.open(img_path).convert('RGB')
        image=loader(image)
        image=image.to(torch.float)
        
        label=img_path.split('\\')[-1].split('-')[1]
        
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image,float(label)
        
    def __len__(self):
        return len(self.images)
        
train_dataset=My_dataset('F:\\age_prediction\\trainset',transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ]),
    # target_transform=ToTensor()
                         )
test_dataset=My_dataset('F:\\age_prediction\\testset',transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()]),
    # target_transform=ToTensor()                    
                        )
        
        
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# train_features, train_labels = next(iter(train_dataloader))


# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

num_epochs=100
num_classes = 6
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16, kernel_size=6, stride=2, padding=2),# the kernel/filter
            nn.Sigmoid(),
            nn.BatchNorm2d(16),
            nn.ReLU(),#activation function
            nn.MaxPool2d(kernel_size=2, stride=2))#pooling layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=6, stride=2, padding=2),
            nn.Sigmoid(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(14*14*32, 1) #out layer
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model=CNN().to(device) 
# criterion = nn.CrossEntropyLoss()
criterion=nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_dataloader)
loss_list_train = []
for epoch in range(num_epochs):
    for i, (images, label) in enumerate(train_dataloader):
        
        images = images.to(device)
        
        label = label.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.float(), label.float())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    # if epoch % eval_step == 0:
    #     eval_loss = eval()
    #     loss_list_val.append(eval_loss)
    #     plt.plot(loss_list_val)
    #     plt.save_fig()
    #     plt.close()
            


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

import torch
import torch.nn   
import torch.optim
import torch.nn.functional 
import torchvision.datasets  
import torchvision.transforms    
import numpy as np
import time



transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
valid = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)  

idx = list(range(len(train)))
np.random.seed(1009)
np.random.shuffle(idx)          
train_idx = idx[:]        

train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)    
train_loader = torch.utils.data.DataLoader(train, batch_size=300, sampler=train_set, num_workers=4)  
test_loader = torch.utils.data.DataLoader(test, num_workers=4, batch_size=1)

class LeNet5(torch.nn.Module):
     
    def __init__(self):   
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)
        self.fc1 = torch.nn.Linear(120, 10)          
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x)) 
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = torch.nn.functional.relu(self.fc1(x))
        return x
     
net = LeNet5()


loss_func = torch.nn.CrossEntropyLoss()  
   
optimization = torch.optim.SGD(net.parameters(), lr = 0.001) 


numEpochs = 20
training_accuracy = []
validation_accuracy = []
start_time = time.time()
for epoch in range(numEpochs):
    
    epoch_training_loss = 0.0
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):        
        inputs, labels = training_batch                              
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)        
        optimization.zero_grad()         
        forward_output = net(inputs)
        loss = loss_func(forward_output, labels)
        loss.backward()   
        optimization.step() 
        epoch_training_loss += loss.item()
        num_batches += 1
    
    print("epoch: ", epoch, ", loss: ", epoch_training_loss/num_batches)            


correct = 0
total = 0
for test_data in test_loader:
    total += 1
    inputs, actual_val = test_data 
    predicted_val = net(torch.autograd.Variable(inputs))   
    max_score, idx = torch.max(predicted_val, 1)
    idx = idx.item()
    correct += (idx == actual_val.item())

print("Total time:", time.time() - start_time)
print("Classifier Accuracy: ", correct/total * 100)

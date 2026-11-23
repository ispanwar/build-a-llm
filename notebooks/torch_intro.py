# Databricks notebook source
print("Hello World!")

# COMMAND ----------

import torch

torch.__version__

# COMMAND ----------

torch.cuda.is_available()

# COMMAND ----------

tensor0d = torch.tensor(1)
print(tensor0d)
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d) 
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor2d)
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor3d)

# COMMAND ----------

print(tensor1d.dtype)

# COMMAND ----------

tensor2d.shape

# COMMAND ----------

print(tensor2d.reshape(3, 2))

# COMMAND ----------

tensor2d_T = tensor2d.view(3,2)
tensor2d_T

# COMMAND ----------

tensor3d.T

# COMMAND ----------

prod = tensor2d.matmul(tensor2d.T)
print(prod)

# COMMAND ----------

print(tensor2d @ tensor2d.T) # matrix multiplication using @ operator

# COMMAND ----------

# Seeing models as computational graphs
import torch.nn.functional as F

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2])
b = torch.tensor([0.0])
z = x1 * w1 + b

a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
print(loss)

# COMMAND ----------

# computing gradients via autograd

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2],requires_grad=True)
b = torch.tensor([0.0],requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = torch.autograd.grad(loss, w1,retain_graph=True)
grad_L_b = torch.autograd.grad(loss, b,retain_graph=True)

print(grad_L_w1)
print(grad_L_b)

# COMMAND ----------

loss.backward()
print(w1.grad)
print(b.grad)

# COMMAND ----------

# implementing a neural net

class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            
            # 1st hidden layer / input layer
            torch.nn.Linear(num_inputs,30),
            torch.nn.ReLU(),
            
            # 2nd hidden layer
            torch.nn.Linear(30,20),
            torch.nn.ReLU(),
            
            # output layer
            torch.nn.Linear(20,num_outputs),
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

# COMMAND ----------

model = NeuralNetwork(num_inputs=50, num_outputs=3)
print(model)

# COMMAND ----------

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# COMMAND ----------

print(model.layers[0].weight)

# COMMAND ----------

print(model.layers[0].weight.shape)

# COMMAND ----------

print(model.layers[0].bias)

# COMMAND ----------

# reproducing the same random numbers and weights
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=50, num_outputs=3)
print(model.layers[0].weight)

# COMMAND ----------

torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)

# COMMAND ----------

# Setting up efficient data loaders

# toy dataset
X_train = torch.tensor([
[-1.2, 3.1],
[-0.9, 2.9],
[-0.5, 2.6],
[2.3, -1.1],
[2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
[-0.8, 2.8],
[2.6, -1.6],
])
y_test = torch.tensor([0, 1])

# COMMAND ----------

# Defining a custom dataset class

from torch.utils.data import Dataset

class ToyDataset(Dataset):
    """A custom dataset class for the toy problem."""

    def __init__(self, X,y):
        self.features = X
        self.labels = y
        
    def __getitem__(self,index): # returning exactly one item from the dataset via index
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        # returning the total number of items in the dataset
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# COMMAND ----------

# instantiating data loaders

from torch.utils.data import DataLoader
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_ds,
    batch_size = 2,
    shuffle = True,
    num_workers = 0
)

test_loader = DataLoader(
    dataset = test_ds,
    batch_size = 2,
    shuffle = False,
    num_workers = 0
)

for idx, (x,y) in enumerate(train_loader):
    print(f"Batch {idx}:")
    print(f"x: {x}")
    print(f"y: {y}")

# COMMAND ----------

# training a neural net

import torch.nn.functional as F
torch.manual_seed(123)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 100

for epoch in range(num_epochs):
    model.train() # setting the model to training mode
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ### Logging
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    model.eval() # setting the model to evaluation mode

# COMMAND ----------

model.eval()

with torch.no_grad():
    outputs = model(X_train)
print(outputs)

# COMMAND ----------

predictions = torch.argmax(outputs, dim=1)
print(predictions)
torch.sum(predictions==y_train)

# COMMAND ----------

# a function for a metric like accuracy

def compute_accuracy(model, data_loader):
    model = model.eval()
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(data_loader):
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = predictions==labels
        correct += torch.sum(compare)
        total_examples += len(compare)
    
    return (correct/total_examples).item()

# COMMAND ----------

print(compute_accuracy(model, train_loader))

# COMMAND ----------

print(compute_accuracy(model, test_loader))

# COMMAND ----------


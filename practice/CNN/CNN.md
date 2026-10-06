# CNN Complete Project (Part 1)

**Project:** Handwritten Digit Classification using MNIST

In this part, we'll cover:

1. Import Libraries
2. Load Dataset
3. Apply Transforms
4. Create DataLoaders
5. Build CNN Model

---

# Step 1: Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
```

---

## Why these libraries?

```python
import torch
```

Main PyTorch library.

Example:

* Create tensors
* Move data to GPU
* Perform mathematical operations

---

```python
import torch.nn as nn
```

Contains all neural network layers.

Examples

```python
nn.Linear()
nn.Conv2d()
nn.MaxPool2d()
nn.ReLU()
nn.Flatten()
nn.CrossEntropyLoss()
```

---

```python
import torch.optim as optim
```

Contains optimizers.

Example

```python
optim.SGD()

optim.Adam()

optim.RMSprop()
```

---

```python
from torchvision import datasets
```

Loads image datasets.

Examples

```python
MNIST

CIFAR10

FashionMNIST

ImageFolder
```

---

```python
from torchvision import transforms
```

Used to preprocess images.

Examples

```python
Resize

Normalize

ToTensor

RandomFlip

RandomRotation
```

---

```python
from torch.utils.data import DataLoader
```

Creates batches.

Instead of

```
60000 images together
```

it creates

```
Batch 1 → 64 images

Batch 2 → 64 images

Batch 3 → 64 images
```

which is much faster.

---

```python
import matplotlib.pyplot as plt
```

Used for displaying images and plotting graphs.

---

# Step 2: Device Configuration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
```

Output

```
cuda
```

or

```
cpu
```

Later we'll send our model and data to this device:

```python
model.to(device)

images = images.to(device)

labels = labels.to(device)
```

---

# Step 3: Hyperparameters

```python
batch_size = 64

learning_rate = 0.001

epochs = 5
```

### What do these mean?

### Batch Size

How many images are processed before updating the weights?

```
64 images

↓

Calculate Loss

↓

Update weights

↓

Next 64 images
```

---

### Learning Rate

Controls how big the weight updates are.

Too high:

```
May overshoot the best solution.
```

Too low:

```
Training becomes very slow.
```

---

### Epochs

One complete pass through the training dataset.

Example:

```
60000 images

↓

Epoch 1

↓

Epoch 2

↓

Epoch 3
```

---

# Step 4: Image Transform

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

### What happens here?

### ToTensor()

Converts

```
Image

↓

Tensor
```

Example

```
28×28 image

↓

Tensor (1×28×28)
```

Pixel values become

```
0-255

↓

0-1
```

---

### Normalize()

Makes training more stable.

Formula

```
(image - mean) / std
```

Here

```
mean = 0.5

std = 0.5
```

So pixel values roughly become

```
[-1 , 1]
```

instead of

```
[0 , 1]
```

---

# Step 5: Download Dataset

```python
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
```

---

### Parameters

```python
root="./data"
```

Dataset is stored inside

```
project/

data/

MNIST/
```

---

```python
train=True
```

Loads

```
60000 images
```

---

```python
train=False
```

Loads

```
10000 images
```

---

```python
download=True
```

Downloads the dataset if it isn't already present.

---

```python
transform=transform
```

Applies

```
ToTensor()

Normalize()
```

to every image.

---

# Step 6: Create DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
```

---

### Why DataLoader?

Instead of

```
60000 images

↓

Model
```

we do

```
64

↓

64

↓

64

↓

...
```

This is faster and uses less memory.

---

### Why shuffle?

Training

```python
shuffle=True
```

Randomizes the order of images every epoch.

Example

Instead of

```
0

1

2

3
```

it becomes

```
8

3

5

0

9
```

This helps the model learn better.

---

Testing

```python
shuffle=False
```

No need to shuffle because we only want to evaluate.

---

# Step 7: Visualize Images (Optional)

```python
images, labels = next(iter(train_loader))

plt.figure(figsize=(10, 3))

for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(labels[i].item())
    plt.axis("off")

plt.show()
```

This displays six sample handwritten digits from the training set.

---

# Step 8: Build CNN Model

```python
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),

            nn.ReLU(),

            nn.Linear(128, 10)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
```

---

# Step 9: Create the Model

```python
model = CNN().to(device)

print(model)
```

This moves the model to the selected device (`cuda` if available, otherwise `cpu`) and prints the model architecture.

---

# Shape Flow Through the Network

```
Input Image
(1 × 28 × 28)

↓

Conv2D (32 filters)

↓

32 × 28 × 28

↓

ReLU

↓

MaxPool

↓

32 × 14 × 14

↓

Conv2D (64 filters)

↓

64 × 14 × 14

↓

ReLU

↓

MaxPool

↓

64 × 7 × 7

↓

Flatten

↓

3136

↓

Linear (128)

↓

Linear (10)

↓

Output (10 classes)
```

---

## End of Part 1 ✅

You now have:

* ✔ All imports
* ✔ Device setup
* ✔ Hyperparameters
* ✔ Dataset loading
* ✔ Image transforms
* ✔ DataLoaders
* ✔ Data visualization
* ✔ Complete CNN model definition

In **Part 2**, we'll add:

* Loss function (`CrossEntropyLoss`)
* Optimizer (`Adam`)
* Complete training loop
* Test loop
* Accuracy calculation
* Loss printing
* Saving the trained model

# CNN Complete Project (Part 2)

**Topic:** Training, Testing, Accuracy, and Saving the Model

In Part 1, we built the CNN model and prepared the data. Now we'll train it and evaluate its performance.

---

# Step 10: Define Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

### Why CrossEntropyLoss?

Our model predicts **10 classes (digits 0–9)**, and **only one class is correct** for each image.

Example:

```
Image: 7

Model Output:

0 : 0.01
1 : 0.02
2 : 0.05
3 : 0.03
4 : 0.02
5 : 0.01
6 : 0.02
7 : 0.80  ✅
8 : 0.03
9 : 0.01
```

`CrossEntropyLoss` compares the prediction with the correct label and tells the model how wrong it is.

---

# Step 11: Define Optimizer

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)
```

### Why Adam?

Adam automatically adjusts the learning rate for each parameter and usually converges faster than plain SGD.

It updates **all trainable weights** in the model.

---

# Step 12: Training Loop

```python
for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )
```

---

# Let's understand every step

---

## Start Training Mode

```python
model.train()
```

Tells PyTorch

> "We are training now."

This is important because some layers (like Dropout and BatchNorm) behave differently during training and testing.

---

## Loop Through Every Batch

```python
for images, labels in train_loader:
```

Suppose:

```
Batch Size = 64
```

Then

```
images

Shape

64 × 1 × 28 × 28
```

and

```
labels

Shape

64
```

---

## Move to GPU/CPU

```python
images = images.to(device)
labels = labels.to(device)
```

If you're using CUDA, both the data and model must be on the GPU.

---

## Clear Old Gradients

```python
optimizer.zero_grad()
```

PyTorch **accumulates gradients by default**.

If we don't clear them, gradients from previous batches would be added, leading to incorrect updates.

---

## Forward Pass

```python
outputs = model(images)
```

This calls:

```python
model.forward(images)
```

The network processes the images and returns predictions.

Example output shape:

```
64 × 10
```

Meaning:

```
64 images

↓

10 scores for each image
```

---

## Compute Loss

```python
loss = criterion(outputs, labels)
```

Compares predictions with the true labels.

If predictions improve:

```
Loss ↓
```

---

## Backward Pass

```python
loss.backward()
```

Computes gradients for every trainable parameter using backpropagation.

---

## Update Weights

```python
optimizer.step()
```

Updates the weights using the gradients.

```
Old Weights

↓

Gradient

↓

New Weights
```

---

## Track Loss

```python
running_loss += loss.item()
```

`loss.item()` converts the tensor loss into a normal Python number.

---

## Print Average Loss

```python
running_loss / len(train_loader)
```

This gives the average loss for the entire epoch.

Example:

```
Epoch 1 Loss: 0.45

Epoch 2 Loss: 0.20

Epoch 3 Loss: 0.09
```

A decreasing loss usually means the model is learning.

---

# Step 13: Testing the Model

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
```

---

# Understanding the Test Loop

---

## Evaluation Mode

```python
model.eval()
```

Switches the model to evaluation mode.

---

## Disable Gradient Calculation

```python
with torch.no_grad():
```

Since we're not training, gradients aren't needed.

Benefits:

* Faster inference
* Lower memory usage

---

## Forward Pass

```python
outputs = model(images)
```

Outputs shape:

```
64 × 10
```

---

## Get Predicted Class

```python
_, predicted = torch.max(outputs, 1)
```

Example:

```
Outputs

[0.01 0.02 0.91 0.01 ...]

↓

Predicted

2
```

The model selects the class with the highest score.

---

## Count Total Samples

```python
total += labels.size(0)
```

If the batch size is 64:

```
total += 64
```

---

## Count Correct Predictions

```python
correct += (predicted == labels).sum().item()
```

Example:

```
Predicted

2 4 7 8

Actual

2 4 5 8
```

Correct predictions:

```
3
```

---

## Compute Accuracy

```python
accuracy = 100 * correct / total
```

Example:

```
9800 / 10000

↓

98%
```

---

# Step 14: Save the Model

```python
torch.save(
    model.state_dict(),
    "cnn_mnist.pth"
)

print("Model Saved Successfully!")
```

This saves only the learned weights.

You can reuse them later without retraining.

---

# Full Training Output Example

```
Epoch [1/5] Loss: 0.4125

Epoch [2/5] Loss: 0.1462

Epoch [3/5] Loss: 0.0894

Epoch [4/5] Loss: 0.0618

Epoch [5/5] Loss: 0.0461

Test Accuracy: 98.65%

Model Saved Successfully!
```

---

# Training Workflow Recap

```
Images
        │
        ▼
Move to Device
        │
        ▼
Zero Gradients
        │
        ▼
Forward Pass
        │
        ▼
Compute Loss
        │
        ▼
Backward Pass
        │
        ▼
Optimizer Step
        │
        ▼
Repeat for All Batches
        │
        ▼
End of Epoch
        │
        ▼
Evaluate on Test Set
        │
        ▼
Compute Accuracy
        │
        ▼
Save Model
```

---

## End of Part 2 ✅

You now have:

* ✔ Loss function (`CrossEntropyLoss`)
* ✔ Optimizer (`Adam`)
* ✔ Complete training loop
* ✔ Complete testing loop
* ✔ Accuracy calculation
* ✔ Model evaluation
* ✔ Model saving

In **Part 3**, we'll cover:

* Loading the saved model
* Predicting on new images
* Visualizing predictions
* Common CNN mistakes
* Tips to improve CNN accuracy
* Practice exercises to deepen your understanding


# CNN Complete Project (Part 3)

## Loading Model • Prediction • Visualization • Complete Workflow • Tips

Congratulations! 🎉 You now have a trained CNN. This final part covers how to **reuse** it without retraining.

---

# Step 15: Load Saved Model

```python
import torch

model = CNN().to(device)

model.load_state_dict(torch.load("cnn_mnist.pth", map_location=device))

model.eval()

print("Model Loaded Successfully!")
```

---

## What happens here?

### Create the model

```python
model = CNN()
```

This creates the same CNN architecture.

At this point,

```
Weights

↓

Random
```

---

### Load weights

```python
model.load_state_dict(...)
```

Now

```
Random Weights

↓

Learned Weights
```

The model is ready to make predictions.

---

# Step 16: Predict One Image

```python
images, labels = next(iter(test_loader))

image = images[0].unsqueeze(0).to(device)

label = labels[0]

with torch.no_grad():

    output = model(image)

    _, prediction = torch.max(output, 1)

print("Actual :", label.item())
print("Predicted :", prediction.item())
```

---

## Why `unsqueeze(0)`?

Before:

```python
images[0].shape

torch.Size([1,28,28])
```

CNN expects

```
Batch

↓

Channels

↓

Height

↓

Width
```

After

```python
unsqueeze(0)
```

Shape becomes

```
torch.Size([1,1,28,28])
```

Now the model accepts it.

---

# Step 17: Display Prediction

```python
plt.imshow(images[0].squeeze(), cmap="gray")

plt.title(f"Prediction : {prediction.item()}")

plt.axis("off")

plt.show()
```

Example

```
Actual

7

Prediction

7
```

---

# Step 18: Predict Multiple Images

```python
images, labels = next(iter(test_loader))

images = images.to(device)

with torch.no_grad():

    outputs = model(images)

    _, predictions = torch.max(outputs,1)

plt.figure(figsize=(12,4))

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(images[i].cpu().squeeze(), cmap="gray")

    plt.title(
        f"P:{predictions[i].item()}\nT:{labels[i].item()}"
    )

    plt.axis("off")

plt.show()
```

Output

```
P:3 T:3

P:5 T:5

P:1 T:1

P:9 T:9
```

---

# Step 19: Complete Workflow

```
Import Libraries

↓

Device

↓

Hyperparameters

↓

Transforms

↓

Dataset

↓

DataLoader

↓

CNN Model

↓

Loss Function

↓

Optimizer

↓

Training

↓

Testing

↓

Accuracy

↓

Save Model

↓

Load Model

↓

Prediction

↓

Display Result
```

This is the same workflow you'll follow for almost every deep learning project.

---

# How CNN Learns

Suppose this image is a cat.

```
Image

↓

Edges

↓

Corners

↓

Eyes

↓

Nose

↓

Face

↓

Cat
```

CNN automatically learns these features.

You **don't manually tell** it where the eyes or ears are.

---

# Why Convolution?

A normal neural network treats an image like:

```
784 numbers
```

It ignores where pixels are located.

CNN says

```
These nearby pixels form an edge.

These edges form an eye.

These eyes form a face.
```

It preserves spatial information.

---

# Why MaxPooling?

Suppose we have

```
8×8 Image
```

Pooling makes it

```
4×4
```

Benefits

* Faster
* Less memory
* Keeps important features
* Reduces overfitting

---

# Why Flatten?

CNN output

```
64 × 7 × 7
```

A Linear layer needs a 1D vector.

Flatten converts

```
64 × 7 × 7

↓

3136
```

---

# Why Fully Connected Layer?

After feature extraction, we need to classify.

Example

```
Edges

↓

Shapes

↓

Digit

↓

Prediction
```

---

# Common Mistakes

### 1. Forgetting `model.train()`

Training may behave incorrectly if layers like Dropout or BatchNorm are used.

---

### 2. Forgetting `model.eval()`

Evaluation results may be inconsistent.

---

### 3. Forgetting `torch.no_grad()`

Inference becomes slower and uses unnecessary memory.

---

### 4. Wrong image size

If the image isn't

```
1×28×28
```

MNIST CNN won't work.

---

### 5. Forgetting `.to(device)`

If the model is on GPU but the images stay on CPU (or vice versa), you'll get a device mismatch error.

---

# How to Improve Accuracy

* Train for more epochs
* Add another convolution layer
* Use Batch Normalization
* Use Dropout
* Tune the learning rate
* Apply data augmentation (for real-world image datasets)
* Use transfer learning on larger datasets

---

# Real-World Uses of CNN

* Face Recognition
* Number Plate Recognition
* Medical Image Analysis
* Object Detection
* Self-Driving Cars
* Satellite Image Classification
* Plant Disease Detection
* OCR (Optical Character Recognition)

---

# Mini Practice Exercises

### Exercise 1

Increase

```python
epochs = 5
```

to

```python
epochs = 10
```

Observe how the loss and accuracy change.

---

### Exercise 2

Change

```python
nn.Conv2d(32,64,3,padding=1)
```

to

```python
nn.Conv2d(32,128,3,padding=1)
```

Update the first Linear layer to match the new output size.

---

### Exercise 3

Replace

```python
optim.Adam(...)
```

with

```python
optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

Compare the training behavior.

---

### Exercise 4

Add

```python
nn.Dropout(0.5)
```

before the final Linear layer and see how it affects training.

---

# Final CNN Architecture

```
Input Image (1×28×28)
          │
          ▼
Conv2D (1 → 32)
          │
          ▼
ReLU
          │
          ▼
MaxPool
          │
          ▼
Conv2D (32 → 64)
          │
          ▼
ReLU
          │
          ▼
MaxPool
          │
          ▼
Flatten
          │
          ▼
Linear (3136 → 128)
          │
          ▼
ReLU
          │
          ▼
Linear (128 → 10)
          │
          ▼
Prediction (0–9)
```

# 🎉 Congratulations!

You now understand the **entire lifecycle of a CNN project**:

* Importing libraries
* Preparing data
* Building the model
* Training
* Evaluating
* Saving
* Loading
* Making predictions
* Improving performance

This workflow will feel very familiar when you move on to **RNN, LSTM, Transformer, and Transfer Learning**—the biggest changes will be the **model architecture** and the **type of data** (text, sequences, or pretrained image models).

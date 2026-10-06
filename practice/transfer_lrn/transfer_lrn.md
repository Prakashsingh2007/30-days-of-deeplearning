Since you're learning **one model at a time**, below is a **complete Transfer Learning project** in **PyTorch** from importing libraries to model evaluation.

This example uses **ResNet18** pretrained on ImageNet and classifies your own images stored in folders.

# Dataset Structure

```
datasets/
└── cats_dogs/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    │
    ├── val/
    │   ├── cats/
    │   └── dogs/
    │
    └── test/
        ├── cats/
        └── dogs/
```

---

# Part 1 - Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models

from torch.utils.data import DataLoader
```

---

# Part 2 - Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
```

---

# Part 3 - Hyperparameters

```python
batch_size = 32
learning_rate = 0.001
epochs = 5

image_size = 224
```

---

# Part 4 - Data Transforms

```python
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
```

---

# Part 5 - Load Dataset

```python
train_dataset = datasets.ImageFolder(
    root="datasets/cats_dogs/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root="datasets/cats_dogs/val",
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root="datasets/cats_dogs/test",
    transform=test_transform
)
```

---

# Part 6 - DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
```

---

# Part 7 - Class Names

```python
print(train_dataset.classes)

num_classes = len(train_dataset.classes)

print(num_classes)
```

---

# Part 8 - Load Pretrained Model

```python
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
```

---

# Part 9 - Freeze Feature Extractor

```python
for param in model.parameters():
    param.requires_grad = False
```

The pretrained convolution layers will not be updated.

---

# Part 10 - Replace Final Layer

```python
in_features = model.fc.in_features

model.fc = nn.Linear(
    in_features,
    num_classes
)
```

---

# Part 11 - Send Model to Device

```python
model = model.to(device)
```

---

# Part 12 - Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

---

# Part 13 - Optimizer

Only train the new classifier.

```python
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=learning_rate
)
```

---

# Part 14 - Training Loop

```python
for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {train_accuracy:.2f}%"
    )
```

---

# Part 15 - Validation

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total

print("Validation Accuracy:", val_accuracy)
```

---

# Part 16 - Test Model

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

test_accuracy = 100 * correct / total

print("Test Accuracy:", test_accuracy)
```

---

# Part 17 - Save Model

```python
torch.save(
    model.state_dict(),
    "models/transfer_learning_resnet18.pth"
)

print("Model Saved")
```

---

# Part 18 - Load Saved Model

```python
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        "models/transfer_learning_resnet18.pth",
        map_location=device
    )
)

model.to(device)

model.eval()
```

---

# Part 19 - Fine-Tuning (Optional)

Instead of training only the final layer, you can unfreeze the last residual block to improve accuracy.

```python
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
```

Now both `layer4` and the new `fc` layer will be trained.

---

# Transfer Learning Workflow

```
Import Libraries
        ↓
Load Dataset
        ↓
Apply Transforms
        ↓
Create DataLoaders
        ↓
Load Pretrained Model
        ↓
Freeze Feature Extractor
        ↓
Replace Final Classifier
        ↓
Loss Function
        ↓
Optimizer
        ↓
Training Loop
        ↓
Validation
        ↓
Testing
        ↓
Save Model
```

### What is Transfer Learning?

Transfer learning is a technique where you start with a model that has already learned useful features from a large dataset (such as ImageNet) and adapt it to a new task. Instead of training from scratch, you reuse the pretrained feature extractor and replace or fine-tune the final classification layer for your own classes.

### Advantages

* Requires much less training data.
* Trains much faster than starting from scratch.
* Usually achieves higher accuracy on small and medium-sized datasets.
* Reduces the risk of overfitting.

### Limitations

* Works best when the new task is similar to the original pretrained task.
* Large pretrained models can require significant memory.
* Fine-tuning too many layers on a small dataset can still lead to overfitting.

This is the standard transfer learning workflow you'll use with pretrained architectures such as **ResNet**, **VGG**, **DenseNet**, **EfficientNet**, **MobileNet**, and **Vision Transformer (ViT)** in PyTorch.

Below is a complete **LSTM model from scratch in PyTorch** using the **MNIST dataset**. It includes everything:

* Import libraries
* Load dataset
* Create DataLoader
* Build LSTM model
* Loss function
* Optimizer
* Training loop
* Testing loop
* Print accuracy

---

# Step 1: Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

---

# Step 2: Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

---

# Step 3: Hyperparameters

```python
input_size = 28        # One row contains 28 pixels
sequence_length = 28   # Total rows

hidden_size = 128
num_layers = 2

num_classes = 10

batch_size = 64

learning_rate = 0.001

epochs = 5
```

---

# Step 4: Load Dataset

```python
transform = transforms.ToTensor()

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

# Step 5: DataLoader

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

# Step 6: Build LSTM Model

```python
class LSTMModel(nn.Module):

    def __init__(self):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        # x shape:
        # (batch_size, 1, 28, 28)

        x = x.squeeze(1)

        # x becomes
        # (batch_size, 28, 28)

        output, (hidden, cell) = self.lstm(x)

        # hidden shape:
        # (num_layers, batch_size, hidden_size)

        out = hidden[-1]

        output = self.fc(out)

        return output
```

---

# Step 7: Create Model

```python
model = LSTMModel().to(device)

print(model)
```

---

# Step 8: Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

---

# Step 9: Optimizer

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)
```

---

# Step 10: Training

```python
for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}"
    )
```

---

# Step 11: Testing

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
```

---

# Step 12: Accuracy

```python
accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
```

---

# Complete Flow

```
Import Libraries
        ↓
Load Dataset
        ↓
Create DataLoader
        ↓
Define Hyperparameters
        ↓
Create LSTM Model
        ↓
Move Model to Device
        ↓
Define Loss Function
        ↓
Define Optimizer
        ↓
Training Loop
    ↓
Forward Pass
    ↓
Calculate Loss
    ↓
Zero Gradients
    ↓
Backward Pass
    ↓
Optimizer Step
        ↓
Testing
        ↓
Calculate Accuracy
        ↓
Print Results
```

## Expected Accuracy

After 5 epochs, you should typically achieve:

* **Training Accuracy:** ~98–99%
* **Test Accuracy:** ~98–99%

This is the standard LSTM implementation using PyTorch's built-in `nn.LSTM`, suitable for sequence modeling tasks like treating each MNIST image as a sequence of 28 rows. If by "from scratch" you meant implementing the LSTM cell equations manually (without using `nn.LSTM`), I can also provide a complete implementation of an `LSTMCell` and stacked LSTM built entirely from basic linear layers and tensor operations.

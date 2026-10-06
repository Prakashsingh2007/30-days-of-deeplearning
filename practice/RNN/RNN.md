Below is a **complete RNN code from scratch** in **PyTorch**, starting from importing libraries, creating data, building the RNN model, training, and testing. This example uses a **toy sequence classification dataset** so you can understand the entire pipeline before using real datasets like IMDB or text files.

---

# Step 1 - Import Libraries

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
```

---

# Step 2 - Hyperparameters

```python
input_size = 1
hidden_size = 32
num_layers = 1
num_classes = 2

sequence_length = 10

batch_size = 16
learning_rate = 0.001
epochs = 20
```

---

# Step 3 - Create Dataset

Here we create random sequences.

If the sum of numbers > 5 → Class 1

Else → Class 0

```python
torch.manual_seed(42)

X = torch.rand(1000, sequence_length, input_size)

y = (X.sum(dim=1) > 5).long().squeeze()
```

Dataset shape

```
X : (1000,10,1)

y : (1000)
```

---

# Step 4 - Train/Test Split

```python
train_size = 800

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]
```

---

# Step 5 - DataLoader

```python
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

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

# Step 6 - Build RNN Model

```python
class RNNModel(nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(
            num_layers,
            x.size(0),
            hidden_size
        )

        out, hidden = self.rnn(x, h0)

        out = out[:, -1, :]

        out = self.fc(out)

        return out
```

---

# Step 7 - Create Model

```python
model = RNNModel()

print(model)
```

Output

```
RNNModel(
  (rnn): RNN(1,32,batch_first=True)
  (fc): Linear(32,2)
)
```

---

# Step 8 - Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

---

# Step 9 - Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)
```

---

# Step 10 - Training Loop

```python
for epoch in range(epochs):

    model.train()

    running_loss = 0

    for sequences, labels in train_loader:

        outputs = model(sequences)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss = {running_loss/len(train_loader):.4f}"
    )
```

---

# Step 11 - Testing

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for sequences, labels in test_loader:

        outputs = model(sequences)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
```

---

# Step 12 - Predict on One Sample

```python
sample = X_test[0].unsqueeze(0)

model.eval()

with torch.no_grad():

    output = model(sample)

    prediction = torch.argmax(output, dim=1)

print("Prediction :", prediction.item())
print("Actual Label:", y_test[0].item())
```

---

# Data Flow

```
Dataset
      │
      ▼
DataLoader
      │
      ▼
Input Shape
(batch, sequence_length, input_size)

Example

(16,10,1)

      │
      ▼
RNN Layer

Output Shape

(16,10,32)

      │
Take Last Time Step

out[:, -1, :]

↓

(16,32)

      │
      ▼
Fully Connected Layer

↓

(16,2)

      │
      ▼
CrossEntropyLoss

↓

Backpropagation

↓

Optimizer
```

---

# Shape Example

Suppose batch size = 16.

```
Input

(16,10,1)

↓

RNN

(16,10,32)

↓

Last timestep

(16,32)

↓

Linear Layer

(16,2)

↓

Predicted Classes

16 labels
```

---

## Workflow to Memorize

```
1. Import libraries

2. Set hyperparameters

3. Create/load sequence dataset

4. Split into train/test

5. Create TensorDataset

6. Create DataLoader

7. Build RNN model
      ├── nn.RNN
      └── nn.Linear

8. Create model

9. Define loss function

10. Define optimizer

11. Train
      ├── Forward
      ├── Loss
      ├── Zero gradients
      ├── Backward
      └── Optimizer step

12. Evaluate

13. Predict on new sequence
```

This is the basic "vanilla RNN" pipeline. Once you're comfortable with it, you can apply the same structure to text, sentiment analysis, time-series forecasting, and other sequence tasks by changing the dataset and preprocessing.

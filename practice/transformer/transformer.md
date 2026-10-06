Below is a **complete Transformer implementation from scratch using PyTorch**, following the same style as the RNN and LSTM examples you've been requesting.

This example performs **sequence classification** on a synthetic dataset. The goal is to teach the Transformer architecture, not achieve high accuracy on a real NLP dataset.

---

# 1. Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
```

---

# 2. Create Dummy Dataset

Suppose:

* Vocabulary size = 100
* Sequence length = 20
* Binary Classification

```python
num_samples = 1000
sequence_length = 20
vocab_size = 100

X = torch.randint(0, vocab_size, (num_samples, sequence_length))
y = torch.randint(0, 2, (num_samples,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

# 3. Positional Encoding

Transformers don't know word order.

We add position information.

```python
class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]
```

---

# 4. Multi-Head Self Attention (Scratch)

```python
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1))

        scores = scores / (self.head_dim ** 0.5)

        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)

        out = out.transpose(1,2).contiguous()

        out = out.view(batch_size, seq_len, self.d_model)

        out = self.fc(out)

        return out
```

---

# 5. Feed Forward Network

```python
class FeedForward(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(d_model, d_model*4),
            nn.ReLU(),

            nn.Linear(d_model*4, d_model)

        )

    def forward(self, x):

        return self.net(x)
```

---

# 6. Transformer Encoder Block

```python
class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        attention = self.attention(x)

        x = self.norm1(
            x + self.dropout(attention)
        )

        ff = self.ff(x)

        x = self.norm2(
            x + self.dropout(ff)
        )

        return x
```

---

# 7. Transformer Model

```python
class TransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        num_classes,
        max_len
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position = PositionalEncoding(
            max_len,
            d_model
        )

        self.layers = nn.ModuleList(

            [
                EncoderBlock(
                    d_model,
                    num_heads
                )

                for _ in range(num_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.position(x)

        for layer in self.layers:

            x = layer(x)

        x = x.transpose(1,2)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x
```

---

# 8. Create Model

```python
model = TransformerClassifier(

    vocab_size=100,
    d_model=64,
    num_heads=8,
    num_layers=2,
    num_classes=2,
    max_len=20

)
```

---

# 9. Loss and Optimizer

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

# 10. Training

```python
epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for X_batch, y_batch in loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += y_batch.size(0)

        correct += (
            predicted == y_batch
        ).sum().item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss:{total_loss/len(loader):.4f} "
        f"Accuracy:{100*correct/total:.2f}%"
    )
```

---

# 11. Prediction

```python
model.eval()

sample = torch.randint(
    0,
    vocab_size,
    (1, sequence_length)
)

with torch.no_grad():

    output = model(sample)

    prediction = torch.argmax(
        output,
        dim=1
    )

print("Prediction:", prediction.item())
```

---

# Architecture Flow

```text
Input Tokens
      │
      ▼
Embedding Layer
      │
      ▼
Positional Encoding
      │
      ▼
Encoder Block 1
      │
      ▼
Encoder Block 2
      │
      ▼
Average Pooling
      │
      ▼
Fully Connected Layer
      │
      ▼
Prediction
```

---

## What this implementation includes

* ✅ Token embedding
* ✅ Positional encoding
* ✅ Scaled dot-product self-attention
* ✅ Multi-head attention
* ✅ Residual connections
* ✅ Layer normalization
* ✅ Feed-forward network
* ✅ Stacked encoder blocks
* ✅ Classification head
* ✅ Full training loop
* ✅ Prediction example

This is a **Transformer Encoder** implementation from scratch, which is the foundation of models like **BERT**. Decoder-only models such as GPT add masked self-attention and autoregressive generation, while encoder-decoder models such as the original Transformer for machine translation combine both an encoder stack and a decoder stack.

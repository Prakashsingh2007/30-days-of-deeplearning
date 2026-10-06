Perfect. Since you're at the end of your 30-day Deep Learning bootcamp, let's summarize the five most important models in the simplest way possible.

---

# 1. CNN (Convolutional Neural Network)

## What is it?

A CNN is a neural network specially designed for **images**.

Instead of looking at every pixel separately, it learns patterns like:

* Edges
* Shapes
* Eyes
* Faces
* Objects

just like humans do.

### Example

Input

🐱 Cat Image

CNN learns

```
Edges
 ↓
Eyes
 ↓
Whiskers
 ↓
Cat
```

---

## Where is it used?

* Image Classification
* Face Recognition
* Medical Images
* Self Driving Cars
* Object Detection

---

## Main Layers

```
Image

↓

Conv2D
(find features)

↓

ReLU
(add non-linearity)

↓

MaxPool
(reduce image size)

↓

Conv

↓

Flatten

↓

Linear Layer

↓

Prediction
```

---

## Advantages

* Best for images
* Learns features automatically
* High accuracy

---

## Disadvantages

* Not good for text
* Needs many images
* Computationally expensive

---

## Complete CNN Code

```python
import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x


model=CNN()
```

---

# 2. RNN (Recurrent Neural Network)

## What is it?

RNN is made for **sequence data**.

It remembers previous information.

Example

```
I
↓

love
↓

deep
↓

learning
```

Every word depends on previous words.

---

## Where is it used?

* Text
* Sentiment Analysis
* Speech
* Time Series

---

## Problem

It forgets long sentences.

Example

```
The boy who was standing near the school because of heavy rain ...

```

By the end, it forgets "boy".

This is called

**Vanishing Gradient Problem**

---

## Advantages

* Handles sequences
* Uses previous information

---

## Disadvantages

* Forgets long-term information
* Slow training

---

## Complete RNN Code

```python
import torch
import torch.nn as nn

class RNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=10,
            hidden_size=20,
            batch_first=True
        )

        self.fc = nn.Linear(20,2)

    def forward(self,x):

        output, hidden = self.rnn(x)

        x = output[:,-1,:]

        x = self.fc(x)

        return x

model = RNNModel()
```

---

# 3. LSTM (Long Short-Term Memory)

## What is it?

LSTM is an improved RNN.

It can remember information for a long time.

Think of it as an RNN with a smarter memory.

---

## Why was LSTM created?

Because RNN forgets.

LSTM decides

* What to remember
* What to forget
* What to output

using gates.

---

## Gates

```
Forget Gate

↓

Input Gate

↓

Output Gate
```

---

## Where is it used?

* Chatbots
* Translation
* Speech Recognition
* Stock Prediction
* Time Series

---

## Advantages

* Remembers long information
* Better than RNN

---

## Disadvantages

* Slower
* More parameters

---

## Complete LSTM Code

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=10,
            hidden_size=32,
            batch_first=True
        )

        self.fc = nn.Linear(32,2)

    def forward(self,x):

        output,(hidden,cell)=self.lstm(x)

        x=output[:,-1,:]

        x=self.fc(x)

        return x

model=LSTMModel()
```

---

# 4. Transformer

## What is it?

Transformer is the model behind modern AI like:

* ChatGPT
* Gemini
* Claude
* Llama

Instead of reading one word at a time, it looks at **all words together** using **Attention**.

---

Example

Sentence

```
The animal didn't cross the road because it was tired.
```

Transformer understands "it" refers to "animal" because it attends to all words.

---

## Main Idea

```
Sentence

↓

Embedding

↓

Positional Encoding

↓

Self Attention

↓

Feed Forward

↓

Prediction
```

---

## Why better than LSTM?

LSTM

```
Word1
↓

Word2
↓

Word3
↓

Word4
```

Transformer

```
Word1 ↔ Word2 ↔ Word3 ↔ Word4

All together
```

---

## Uses

* ChatGPT
* Translation
* Summarization
* Coding AI
* Vision Transformer

---

## Advantages

* Fast training
* Better context
* Handles long text

---

## Disadvantages

* Needs lots of data
* Expensive to train

---

## Complete Transformer Code

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(1000,64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64,2)

    def forward(self,x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x[:,-1,:]

        x = self.fc(x)

        return x

model = TransformerModel()
```

---

# 5. Transfer Learning

## What is it?

Instead of training from scratch, use a model that is already trained on millions of images (e.g., ImageNet) and adapt it to your own task.

Example:

A pretrained model already knows:

* Edges
* Shapes
* Animals
* Vehicles

You only teach it to recognize your specific classes (e.g., cats vs. dogs or healthy vs. diseased leaves).

---

## Why use it?

Training from scratch:

* Needs lots of data
* Takes a long time

Transfer Learning:

* Needs less data
* Trains faster
* Often gives better accuracy

---

## Common pretrained models

* ResNet
* VGG
* EfficientNet
* MobileNet
* Vision Transformer (ViT)

---

## Advantages

* High accuracy with small datasets
* Fast training
* Uses powerful pretrained features

---

## Disadvantages

* Limited by the pretrained model
* May not work well if your data is very different from the original training data

---

## Complete Transfer Learning Code (ResNet18)

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier
model.fc = nn.Linear(model.fc.in_features, 2)

# Unfreeze the final layer
for param in model.fc.parameters():
    param.requires_grad = True

print(model)
```

---

# When should you use each model?

| Model                 | Best For                          | Example                                                            |
| --------------------- | --------------------------------- | ------------------------------------------------------------------ |
| **CNN**               | Images                            | Cat vs Dog classification, face recognition                        |
| **RNN**               | Short sequences                   | Basic text classification, simple time series                      |
| **LSTM**              | Long sequences                    | Translation, speech recognition, stock prediction                  |
| **Transformer**       | Modern NLP and long-context tasks | ChatGPT, Gemini, Claude, document understanding                    |
| **Transfer Learning** | Small image datasets              | Medical imaging, plant disease detection, custom image classifiers |

## One-line summary to remember

* **CNN:** Finds patterns in **images**.
* **RNN:** Processes **sequences** by remembering previous steps.
* **LSTM:** An RNN with **better long-term memory**.
* **Transformer:** Uses **attention** to process all sequence elements together.
* **Transfer Learning:** Reuses knowledge from a **pretrained model** to solve a new task quickly.

With these concepts, you've covered the core deep learning architectures you'll encounter most often. The next step is applying them to real-world projects rather than learning many more model types.

# 📉 Pooling Layers in Convolutional Neural Networks (CNNs)

Pooling layers are essential components of CNNs that reduce the spatial dimensions (height & width) of the input feature maps. This helps decrease the number of parameters, reduce computation, and control overfitting while retaining key spatial features.

---

## 🧠 What is Pooling?

> Pooling is a **downsampling operation** applied after convolutional layers.

It works by summarizing the information in small patches (e.g., 2×2) of the input.

---

## 🧪 Common Pooling Types

### 1. **Max Pooling**
- Takes the **maximum** value in each patch.
- Retains the most prominent features.
- Very common in CNNs.

### 2. **Average Pooling**
- Takes the **average** of values in each patch.
- Smoother output; less aggressive than max pooling.


---

## ⚙️ Key Parameters

| Parameter     | Description                                 |
|---------------|---------------------------------------------|
| `pool_size`   | Size of the pooling window (e.g., 2×2)       |
| `strides`     | Steps the window moves (usually same as size)|
| `padding`     | Whether to pad input (same/valid)            |

---

## 🔧 Example in Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
])



# Code Explanations: Labs 1, 2, and 3

This document provides a line-by-line explanation of the code in each of the three Jupyter Notebooks. It explains what each block does, the purpose of each function, and why specific settings are used.

---

## 🧠 Lab 1: Simulate a Perceptron using NumPy
**File**: [ANN_lab1.ipynb](./ANN_lab1.ipynb)

This script builds a single Perceptron from scratch using only raw Python and **NumPy**. It uses a sharp step activation function and learns the weights dynamically through training.

### Full Code Code Block
```python
import numpy as np

# Step Activation Function
def step_function(x):
    if x >= 0:
        return 1
    return 0

# Perceptron Class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    # Predict output
    def predict(self, x):
        total = np.dot(x, self.weights) + self.bias
        return step_function(total)

    # Train the perceptron
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction

                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Training data for AND Gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Create and train perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)

# Display predictions
print("Predictions:")
for sample in X:
    print("Input:", sample, "Output:", perceptron.predict(sample))
```

### Detailed Code Walkthrough

1. **`import numpy as np`**
   - Imports the NumPy library, which provides support for multi-dimensional arrays and fast mathematical operations.
2. **`def step_function(x):`**
   - Defines the activation function. It acts as a hard ON/OFF switch. If the net input score `x` is greater than or equal to `0`, it returns `1` (ON). If it is negative, it returns `0` (OFF).
3. **`class Perceptron:`**
   - Defines the class structure for our single-cell network.
4. **`def __init__(self, input_size, learning_rate=0.1):`**
   - The constructor initializes the Perceptron's state:
     - `self.weights = np.zeros(input_size)`: Starts weights at zero for all inputs (e.g., `[0.0, 0.0]`).
     - `self.bias = 0`: Starts the bias threshold at 0.
     - `self.learning_rate = learning_rate`: Sets how fast the model adjusts its weights (set to `0.1` by default).
5. **`def predict(self, x):`**
   - Calculates the dot product of the input array `x` and the weights (`x * weights`), adds the bias, and sends the total score to the `step_function` to get a `0` or `1` output.
6. **`def train(self, X, y, epochs=10):`**
   - This is the learning loop:
     - Runs through the whole dataset `epochs` times (set to 10 rounds).
     - Loops over each input sample and its target label using `zip(X, y)`.
     - Calculates the prediction error: `error = target - prediction`.
     - If `error` is not `0` (i.e. prediction was wrong), it updates the weights and bias:
       - `self.weights += self.learning_rate * error * inputs`
       - `self.bias += self.learning_rate * error`
7. **`X = np.array([...])` and `y = np.array([0, 0, 0, 1])`**
   - Defines the inputs and targets for the **logical AND gate**. The target `y` is only `1` when both inputs are `1`.
8. **`perceptron = Perceptron(input_size=2)`**
   - Creates a Perceptron instance with 2 input features (representing x1 and x2).
9. **`perceptron.train(X, y, epochs=10)`**
   - Runs the training loop for 10 rounds to adjust the weights and bias.
10. **`print(...)` and inference loop**
    - Feeds each of the four AND inputs back into the trained model and prints the final outputs to verify that the Perceptron has successfully learned the AND function.

---

## 🔢 Lab 2: Single Neuron Forward Pass (Sigmoid)
**File**: [ANN_lab2.ipynb](./ANN_lab2.ipynb)

This script demonstrates **forward propagation** manually through a single neuron. Instead of training or updating weights, we hardcode the weights and bias to trace how a neuron computes outputs using a smooth **Sigmoid** activation function.

### Full Code Code Block
```python
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input values
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Initial weights and bias
weights = np.array([0.5, 0.5])
bias = -0.7

print("Single Neuron Model Output:\n")

for x in X:
    # Weighted sum
    z = np.dot(x, weights) + bias

    # Apply sigmoid activation
    output = sigmoid(z)

    # Convert to binary class
    prediction = 1 if output >= 0.5 else 0

    print(f"Input: {x}, Weighted Sum = {z:.2f}, Output = {output:.4f}, Class = {prediction}")
```

### Detailed Code Walkthrough

1. **`import numpy as np`**
   - Imports NumPy for array structures and vectorized mathematical functions.
2. **`def sigmoid(x):`**
   - Implements the sigmoid activation function: sigmoid(x) = 1 / (1 + e^-x).
   - `np.exp(-x)` computes e^-x for the input value. The sigmoid squashes any score into a continuous range between 0 and 1, representing confidence/probability.
3. **`X = np.array([...])`**
   - Sets up the four combinations of inputs for the AND gate dataset.
4. **`weights = np.array([0.5, 0.5])` and `bias = -0.7`**
   - Manually defines fixed, preset settings. These weights and bias are chosen specifically so that the single neuron can solve the AND gate without any training.
5. **`for x in X:`**
   - Starts a loop to feed each of the four input samples through the neuron.
6. **`z = np.dot(x, weights) + bias`**
   - Computes the weighted sum of inputs and weights, then adds the bias:
     - For `[0, 0]`, z = (0 * 0.5) + (0 * 0.5) - 0.7 = -0.7.
     - For `[1, 1]`, z = (1 * 0.5) + (1 * 0.5) - 0.7 = 0.3.
7. **`output = sigmoid(z)`**
   - Passes the score `z` to the sigmoid function to get a decimal output (probability). For example, `sigmoid(0.3)` is approximately `0.5744`.
8. **`prediction = 1 if output >= 0.5 else 0`**
   - Applies thresholding: if the probability is `0.5` or higher, the class prediction is `1` (YES). If it is less than `0.5`, the prediction is `0` (NO).
9. **`print(f"...")`**
   - Prints the formatted inputs, weighted sum, sigmoid probability, and final class prediction.

---

## ⚡ Lab 3: Feedforward Neural Network (TensorFlow/Keras)
**File**: [ANN_lab3.ipynb](./ANN_lab3.ipynb)

This script builds, compiles, and trains a multi-layer neural network (a hidden layer of 4 neurons + 1 output neuron) using **TensorFlow** and **Keras** to solve the logical AND gate.

### Full Code Code Block
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# AND Gate Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)

y = np.array([[0],
              [0],
              [0],
              [1]], dtype=np.float32)

# Feedforward Neural Network
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X, y, epochs=500, verbose=0)

# Predictions
print("Predictions:")
predictions = model.predict(X)

for i in range(len(X)):
    print(f"Input: {X[i]} => Predicted: {predictions[i][0]:.4f} => Class: {int(predictions[i][0] >= 0.5)}")
```

### Detailed Code Walkthrough

1. **Imports (`numpy`, `tensorflow`, `Sequential`, `Input`, `Dense`)**
   - `numpy`: For dataset structure.
   - `tensorflow`: The core backend deep learning framework.
   - `Sequential`: A Keras model class that lets us stack layers sequentially (one after another).
   - `Input`, `Dense`: Keras layers. `Input` specifies the input size, and `Dense` defines a fully connected layer (where every neuron connects to all inputs from the previous layer).
2. **`X` and `y` Arrays with `dtype=np.float32`**
   - Defines inputs and targets for the AND gate.
   - We specify `float32` instead of the default 64-bit float because deep learning frameworks like TensorFlow run faster and consume less memory with 32-bit floats.
3. **`model = Sequential([...])`**
   - Constructs the network architecture:
     - `Input(shape=(2,))`: Tells the model that each input sample has 2 features.
     - `Dense(4, activation='relu')`: The hidden layer with 4 neurons. It uses **ReLU** (Rectified Linear Unit) activation, which filters out negative scores (turning them to 0) and passes positive scores through.
     - `Dense(1, activation='sigmoid')`: The output layer with 1 neuron. It uses **Sigmoid** activation to return a probability value between `0` and `1`.
4. **`model.compile(...)`**
   - Configures the model before training:
     - `optimizer='adam'`: The Adam optimizer, which updates weights dynamically using running momentum.
     - `loss='binary_crossentropy'`: The loss function for binary (yes/no) classification, which penalizes incorrect confident predictions.
     - `metrics=['accuracy']`: Tells Keras to log classification accuracy during training.
5. **`model.fit(X, y, epochs=500, verbose=0)`**
   - Trains the model by running the dataset through it 500 times (`epochs`).
   - `verbose=0`: Suppresses print logs for each epoch so the console output remains clean.
6. **`predictions = model.predict(X)`**
   - Feeds the input data `X` back into the trained model to perform inference. This returns raw float probability values.
7. **Inference Loop & Thresholding**
   - Loops over predictions and converts the float probability to a binary class: `int(predictions[i][0] >= 0.5)`. Prints the result for each input sample.

---

## 🌸 Lab 4: Multiclass Classification using Keras Multi-Layer Perceptron (MLP)
**File**: [ANN_lab4.ipynb](./ANN_lab4.ipynb)

This script builds, compiles, and evaluates a Multi-Layer Perceptron (MLP) for **Multiclass Classification** on the 3-class **Iris dataset** using **TensorFlow/Keras** and **scikit-learn**.

### Full Code Code Block
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris Dataset (4 features, 3 target classes)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize feature values (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. One-hot encode multiclass target labels
y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)

# 5. Build Keras MLP Model for Multiclass Classification
model = Sequential([
    Input(shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

# 6. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Train the model
print("Training Keras MLP Model on Iris Dataset...")
model.fit(X_train, y_train_encoded, epochs=100, batch_size=16, validation_split=0.1, verbose=0)

# 8. Evaluate model performance on test set
loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

# 9. Predict and decode multiclass predictions using Argmax
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

print("Sample Predictions on Test Set:")
print("-" * 65)
print(f"{'Sample #':<10} | {'True Class':<15} | {'Predicted Class':<15} | {'Probabilities (Setosa, Versicolor, Virginica)'}")
print("-" * 65)

for i in range(10):
    sample_probs = [f"{p:.3f}" for p in predictions[i]]
    print(f"{i+1:<10} | {target_names[y_test[i]]:<15} | {target_names[predicted_classes[i]]:<15} | {sample_probs}")
```

### Detailed Code Walkthrough

1. **Imports (`load_iris`, `train_test_split`, `StandardScaler`, `to_categorical`)**
   - `load_iris`: Loads the classic Iris flower dataset containing 150 samples, 4 physical features (sepal length/width, petal length/width), and 3 species target classes (Setosa, Versicolor, Virginica).
   - `train_test_split`: Splits data into training (80%) and test (20%) sets while preserving class distribution ratios (`stratify=y`).
   - `StandardScaler`: Scales input features so they have a mean of 0 and a standard deviation of 1. Standardizing features is essential for smooth gradient descent.
   - `to_categorical`: Converts integer labels (`0, 1, 2`) into one-hot encoded binary vectors (`[1,0,0]`, `[0,1,0]`, `[0,0,1]`).
2. **Loading & Dataset Splitting**
   - `X` contains the 150x4 feature matrix, and `y` contains target values `[0, 1, 2]`.
   - `train_test_split` creates `X_train` (120 samples) and `X_test` (30 samples).
3. **Feature Scaling (`StandardScaler`)**
   - `scaler.fit_transform(X_train)` learns the mean and variance of `X_train` and normalizes it.
   - `scaler.transform(X_test)` applies the exact same mean and variance transformation to `X_test` without data leakage.
4. **One-Hot Encoding (`to_categorical`)**
   - Transforms 1D class array `[0, 1, 2]` into 2D one-hot target matrix of shape `(N, 3)` required by `categorical_crossentropy`.
5. **Model Architecture (`Sequential`)**
   - `Input(shape=(4,))`: 4 input slots for the scaled features.
   - `Dense(8, activation='relu')`: Hidden layer of 8 neurons with ReLU activation for non-linear pattern extraction.
   - `Dense(3, activation='softmax')`: Output layer with 3 neurons matching the 3 classes. Uses **Softmax** activation to output a probability distribution over the 3 classes that sums to 1.0.
6. **Compilation (`categorical_crossentropy` & `adam`)**
   - Loss function: `categorical_crossentropy`, which evaluates multiclass probability distributions against one-hot targets.
   - Optimizer: `adam` for adaptive weight updates.
   - Metric: `accuracy`.
7. **Model Training (`model.fit`)**
   - Fits the model over 100 `epochs` with a batch size of 16 and a 10% validation split.
8. **Evaluation (`model.evaluate`)**
   - Evaluates test loss and test accuracy on the 30 unseen test samples.
9. **Predictions & Decoding (`np.argmax`)**
   - `model.predict(X_test)` returns a 3-element probability vector for each sample.
   - `np.argmax(predictions, axis=1)` extracts the index of the highest probability value to give the predicted class integer `(0, 1, or 2)`.
   - Maps predicted integers back to species names (`Setosa`, `Versicolor`, `Virginica`) and prints formatted predictions.


# Artificial Neural Networks (ANN) & Deep Learning Lab

## Conceptual Reference Guide: Labs 1, 2, 3, and 4

This repository contains Jupyter Notebooks that guide you through building and understanding neural networks. We start with a basic Perceptron built from scratch, move to a single neuron with a smooth activation function, build a binary Feedforward Neural Network using TensorFlow/Keras, and culminate with a Multi-Layer Perceptron (MLP) for Multiclass Classification.

- **Lab 1 Notebook**: [ANN_lab1.ipynb](./ANN_lab1.ipynb) — Perceptron from scratch using NumPy (AND Gate).
- **Lab 2 Notebook**: [ANN_lab2.ipynb](./ANN_lab2.ipynb) — Single neuron forward pass with Sigmoid activation (AND Gate).
- **Lab 3 Notebook**: [ANN_lab3.ipynb](./ANN_lab3.ipynb) — Multi-Layer Perceptron using TensorFlow/Keras (Binary AND Gate).
- **Lab 4 Notebook**: [ANN_lab4.ipynb](./ANN_lab4.ipynb) — Keras MLP for Multiclass Classification (Iris Dataset).

> [!NOTE]
> For a very simple version with stories and analogies, check out the [Explaination Guide](./explaination.md).
>
> [!TIP]
> For a line-by-line detailed explanation of the code in each notebook, check out the [Code Explanations Guide](./codex.md).

---

## 📌 Table of Contents

1. [Notebook Overview](#-notebook-overview)
2. [Lab 1: Perceptron from Scratch (NumPy)](#-lab-1-perceptron-from-scratch-numpy)
   - [Conceptual Foundations](#lab-1-conceptual-foundations)
   - [How It Learns](#lab-1-how-it-learns)
   - [Code Breakdown](#lab-1-code-breakdown)
3. [Lab 2: Single Neuron Forward Pass (Sigmoid)](#-lab-2-single-neuron-forward-pass-sigmoid)
   - [Conceptual Foundations](#lab-2-conceptual-foundations)
   - [Step-by-Step Hand Calculations](#lab-2-step-by-step-hand-calculations)
   - [Code Breakdown](#lab-2-code-breakdown)
4. [Lab 3: Feedforward Neural Network (TensorFlow/Keras)](#-lab-3-feedforward-neural-network-tensorflowkeras)
   - [Conceptual Foundations](#lab-3-conceptual-foundations)
   - [Layers & Trainable Parameters](#lab-3-layers--trainable-parameters)
   - [Loss & Optimizers Made Simple](#lab-3-loss--optimizers-made-simple)
   - [Code Breakdown](#lab-3-code-breakdown)
5. [Lab 4: Keras MLP for Multiclass Classification](#-lab-4-keras-mlp-for-multiclass-classification)
   - [Conceptual Foundations](#lab-4-conceptual-foundations)
   - [Layers & Trainable Parameters](#lab-4-layers--trainable-parameters)
   - [Loss & Activation Functions Made Simple](#lab-4-loss--activation-functions-made-simple)
   - [Code Breakdown](#lab-4-code-breakdown)
6. [🎓 The Viva Q&A Guide (25+ Conceptual Questions)](#-the-viva-qa-guide-25-conceptual-questions)

---

## 🗺️ Notebook Overview

The four labs show the complete progression of neural network concepts:

```mermaid
graph TD
    A["Repository Notebooks<br>(Labs 1, 2, 3, and 4)"] --> B[Lab 1: Perceptron from Scratch]
    A --> C[Lab 2: Manual Neuron Forward Pass]
    A --> D[Lab 3: Keras Binary MLP]
    A --> E[Lab 4: Keras Multiclass MLP]

    B --> B1[Uses NumPy only]
    B --> B2[Uses ON/OFF Step Switch]
    B --> B3[Learns AND gate rules]

    C --> C1[Uses fixed weights & bias]
    C --> C2[Uses smooth Sigmoid activation]
    C --> C3[Shows math step-by-step]

    D --> D1[Uses TensorFlow & Keras sequential API]
    D --> D2[Hidden layer with ReLU activation]
    D --> D3[Binary Cross-Entropy & Adam]

    E --> E1[Multi-class dataset - Iris]
    E --> E2[Feature scaling & One-Hot Encoding]
    E --> E3[Softmax activation & Categorical Cross-Entropy]
```

---

## 🧠 Lab 1: Perceptron from Scratch (NumPy)

### Lab 1: Conceptual Foundations

A **Perceptron** is the simplest form of a neural network, introduced by Frank Rosenblatt in 1958. It is a **linear classifier**, meaning it separates data using a single straight line.

- **Inputs**: The data coordinates or features (for the AND gate, these are two binary inputs: `0` or `1`).
- **Weights**: The importance or strength given to each input.
- **Bias**: The threshold or "natural grumpiness" of the neuron. It decides how easy or hard it is for the neuron to activate.
- **Step Activation Function**: A sharp ON/OFF switch. If the net score (inputs multiplied by weights, plus bias) is 0 or positive, it outputs `1`. If the score is negative, it outputs `0`.

**Limitation**: A Perceptron can only solve problems that are **linearly separable** (where a single straight line can separate the two classes). Logical AND, OR, and NAND gates can be separated by a line. Logical XOR and XNOR gates cannot, so a single Perceptron fails to solve them.

### Lab 1: How It Learns

1. **Weighted Sum**: The Perceptron multiplies each input by its weight and adds the bias to get a single score.
2. **Activation**: The score is passed through the ON/OFF Step function to get a predicted class (`0` or `1`).
3. **Weight Update Rule**: If the prediction is wrong, we calculate the error (`target - prediction`). We then adjust the weights and bias by a small step size called the **learning rate** (often written as `0.1` or `0.01`). If the prediction is correct, no changes are made.

### Lab 1: Code Breakdown

- **Step Function**:

  ```python
  def step_function(x):
      if x >= 0:
          return 1
      return 0
  ```

  Returns `1` if the net input is zero or positive, else returns `0`.

- **Constructor (`__init__`)**:

  ```python
  def __init__(self, input_size, learning_rate=0.1):
      self.weights = np.zeros(input_size)
      self.bias = 0
      self.learning_rate = learning_rate
  ```

  Sets up the Perceptron. Weights and bias start at zero. The learning rate is set to `0.1` so weight updates are done in small, controlled steps.

- **Prediction (`predict`)**:

  ```python
  def predict(self, x):
      total = np.dot(x, self.weights) + self.bias
      return step_function(total)
  ```

  Multiplies inputs by weights (dot product), adds bias, and passes the total to the ON/OFF switch.

- **Training (`train`)**:
  ```python
  def train(self, X, y, epochs=10):
      for _ in range(epochs):
          for inputs, target in zip(X, y):
              prediction = self.predict(inputs)
              error = target - prediction
              self.weights += self.learning_rate * error * inputs
              self.bias += self.learning_rate * error
  ```
  Runs through the dataset for a set number of rounds (`epochs`). For each sample, it checks the prediction, calculates the error, and adjusts weights and bias if the prediction was wrong.

---

## 🔢 Lab 2: Single Neuron Forward Pass

### Lab 2: Conceptual Foundations

This lab demonstrates **forward propagation** (how data flows forward through a network) using fixed settings. Instead of training the neuron, we hardcode the weights and bias to show how the math works.

- **Sigmoid Activation**: A smooth "dimmer switch" instead of a sharp ON/OFF step function. It outputs a decimal value between `0` and `1`. This output represents the probability or confidence of the classification (e.g., `0.57` means "57% confident the answer is 1").
- **Thresholding**: To make a final binary prediction, we check if the sigmoid output is `0.5` or higher. If it is, we classify it as `1`; otherwise, it is `0`.

### Lab 2: Step-by-Step Hand Calculations

Using these fixed parameters from the lab:

- Weights: `[0.5, 0.5]`
- Bias: `-0.7`

Here is how the neuron processes each input of the AND gate:

| Input (x1, x2) | Weighted Sum (z) | Sigmoid Formula Output | Final Prediction (Is Output >= 0.5?) |
| :--- | :--- | :--- | :--- |
| **[0, 0]** | (0 * 0.5) + (0 * 0.5) - 0.7 = -0.7 | 1 / (1 + e^0.7) = 0.3318 | **0** (since 0.3318 < 0.5) |
| **[0, 1]** | (0 * 0.5) + (1 * 0.5) - 0.7 = -0.2 | 1 / (1 + e^0.2) = 0.4502 | **0** (since 0.4502 < 0.5) |
| **[1, 0]** | (1 * 0.5) + (0 * 0.5) - 0.7 = -0.2 | 1 / (1 + e^0.2) = 0.4502 | **0** (since 0.4502 < 0.5) |
| **[1, 1]** | (1 * 0.5) + (1 * 0.5) - 0.7 = 0.3 | 1 / (1 + e^-0.3) = 0.5744 | **1** (since 0.5744 >= 0.5) |

### Lab 2: Code Breakdown

- **Sigmoid Activation**:

  ```python
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  ```

  Applies the mathematical sigmoid formula to squash the score into a decimal range between 0 and 1.

- **Inference Loop**:

  ```python
  weights = np.array([0.5, 0.5])
  bias = -0.7

  for x in X:
      z = np.dot(x, weights) + bias
      output = sigmoid(z)
      prediction = 1 if output >= 0.5 else 0
      print(f"Input: {x}, Output = {output:.4f}, Class = {prediction}")
  ```

  Runs the calculations shown in our table and prints out the probability and binary class for each input combination.

---

## ⚡ Lab 3: Feedforward Neural Network (TensorFlow)

### Lab 3: Conceptual Foundations

This lab uses the **TensorFlow and Keras** libraries to build a complete Multi-Layer Perceptron (MLP). Rather than a single neuron, we stack multiple neurons together to form a full network.

- **Input Layer**: Where the input signals enter.
- **Hidden Layer**: A middle layer of neurons that allows the network to learn more complex relationships and patterns.
- **Output Layer**: The final layer that produces the prediction.
- **Backpropagation**: The training algorithm. When the network makes a mistake at the output layer, it passes the error backward through the layers to adjust all the weights and biases.

### Lab 3: Layers & Trainable Parameters

Our network architecture is built as follows:

```
       [Input 1] -------\      /---> [Hidden Neuron 1 (ReLU)] ---\
                         \    /----> [Hidden Neuron 2 (ReLU)] ----\
                          \  /-----> [Hidden Neuron 3 (ReLU)] -----\
       [Input 2] ----------X-------> [Hidden Neuron 4 (ReLU)] ------+---> [Output (Sigmoid)]
```

#### Trainable Parameters Calculation (Common Viva Question):

Trainable parameters are the individual weights and biases that the network learns.

* **Input Layer to Hidden Layer**:
  - Weights: 2 inputs * 4 hidden neurons = 8 weights.
  - Biases: 1 bias per hidden neuron = 4 biases.
  - Subtotal = 8 + 4 = 12 parameters.
* **Hidden Layer to Output Layer**:
  - Weights: 4 hidden neurons * 1 output neuron = 4 weights.
  - Biases: 1 bias per output neuron = 1 bias.
  - Subtotal = 4 + 1 = 5 parameters.
* **Total Parameters**: 12 + 5 = 17 trainable parameters.

### Lab 3: Loss & Optimizers Made Simple

1. **Binary Cross-Entropy Loss (The Scorer)**:
   This is our grade or penalty system. Since this is a binary classifier (yes/no), the loss function measures how close the predicted probability is to the true label. Confident mistakes are penalized heavily, while correct, confident answers get a near-zero penalty.
2. **Adam Optimizer (The Teacher)**:
   A smart learning assistant that automatically adjusts the network's weights based on the loss. It acts dynamically, slowing down or speeding up updates as needed so the model converges quickly and efficiently.

### Lab 3: Code Breakdown

- **Defining the Network**:

  ```python
  model = Sequential([
      Input(shape=(2,)),
      Dense(4, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  ```

  - `Sequential`: Builds the network layer-by-layer.
  - `Input(shape=(2,))`: Tells the model to expect 2 inputs per sample.
  - `Dense(4, activation='relu')`: Creates a hidden layer of 4 fully connected neurons using the **ReLU** activation function.
  - `Dense(1, activation='sigmoid')`: Creates an output layer with 1 neuron using the **Sigmoid** activation function to output a probability between 0 and 1.

- **Compiling the Model**:

  ```python
  model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy']
  )
  ```

- **Training the Model**:
  ```python
  model.fit(X, y, epochs=500, verbose=0)
  ```

---

## 🌸 Lab 4: Keras MLP for Multiclass Classification

### Lab 4: Conceptual Foundations

In contrast to binary classification (where the target is 0 or 1), **Multiclass Classification** involves classifying inputs into one of three or more distinct categories.

In Lab 4, we use the classic **Iris Dataset** (150 flower samples, 4 physical features) to classify flowers into **3 species**:
1. `0`: Iris-Setosa
2. `1`: Iris-Versicolor
3. `2`: Iris-Virginica

Key pipeline steps for Multiclass Classification:
- **Feature Scaling (`StandardScaler`)**: Normalizes input features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$), preventing features with larger scales from dominating weight updates.
- **One-Hot Encoding (`to_categorical`)**: Converts integer targets `[0, 1, 2]` into 3D binary vectors:
  - Class `0` $\rightarrow$ `[1, 0, 0]`
  - Class `1` $\rightarrow$ `[0, 1, 0]`
  - Class `2` $\rightarrow$ `[0, 0, 1]`
- **Softmax Activation**: Applied at the output layer to convert raw output scores (logits) into a normalized probability distribution that sums up to 1.0.

### Lab 4: Layers & Trainable Parameters

Our network architecture for multiclass classification is:

```
    [Sepal Length] ----\
    [Sepal Width]  -----\=====> [Hidden Layer: 8 Neurons (ReLU)] =====> [Output Layer: 3 Neurons (Softmax)]
    [Petal Length] -----/                                                      |
    [Petal Width]  ----/                                                [Probability Vector: Setosa, Versicolor, Virginica]
```

#### Trainable Parameters Calculation:

* **Input Layer to Hidden Layer**:
  - Inputs = 4 features
  - Hidden Neurons = 8
  - Weights = $4 \times 8 = 32$
  - Biases = 8
  - Subtotal = $32 + 8 = 40$ parameters.
* **Hidden Layer to Output Layer**:
  - Hidden Neurons = 8
  - Output Neurons = 3 (1 for each class)
  - Weights = $8 \times 3 = 24$
  - Biases = 3
  - Subtotal = $24 + 3 = 27$ parameters.
* **Total Parameters**: $40 + 27 = 67$ trainable parameters.

### Lab 4: Loss & Activation Functions Made Simple

1. **Softmax Activation Function**:
   The Softmax function squashes a vector of $K$ real values into a probability distribution of $K$ probabilities proportional to the exponentials of the input numbers:
   $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
   This ensures that all outputs are between $0$ and $1$, and their total sum is exactly $1.0$.

2. **Categorical Cross-Entropy Loss**:
   Measures the performance of a classification model whose output is a probability value between 0 and 1. The loss increases as the predicted probability diverges from the actual label:
   $$\text{Loss} = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$
   Where $y_i$ is the true binary indicator (0 or 1) from one-hot encoding, and $\hat{y}_i$ is the predicted probability from Softmax.

3. **Decoding Predictions with `Argmax`**:
   To convert the continuous 3-element probability vector back into a single class prediction, we take the index of the maximum probability using `np.argmax(predictions, axis=1)`.

### Lab 4: Code Breakdown

- **Data Loading & Preprocessing**:
  ```python
  iris = load_iris()
  X, y = iris.data, iris.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  y_train_encoded = to_categorical(y_train, num_classes=3)
  y_test_encoded = to_categorical(y_test, num_classes=3)
  ```

- **Building the Keras Model**:
  ```python
  model = Sequential([
      Input(shape=(4,)),
      Dense(8, activation='relu'),
      Dense(3, activation='softmax')
  ])
  ```

- **Compilation & Training**:
  ```python
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train_encoded, epochs=100, batch_size=16, validation_split=0.1, verbose=0)
  ```

- **Evaluation & Class Decoding**:
  ```python
  predictions = model.predict(X_test, verbose=0)
  predicted_classes = np.argmax(predictions, axis=1)
  ```

---

## 🎓 The Viva Q&A Guide (25+ Conceptual Questions)

### Q1: What is the main objective of these four labs?

**Answer:** They show the evolution of neural networks.

1. **Lab 1** builds a single basic Perceptron from scratch to show the simplest binary classifier.
2. **Lab 2** shows how a single neuron performs a forward pass with a smooth activation function (Sigmoid).
3. **Lab 3** builds a multi-layer binary network with a hidden layer and backpropagation using Keras.
4. **Lab 4** extends multi-layer networks to **Multiclass Classification** using Softmax activation, feature scaling, and one-hot encoding.

### Q2: What is a Perceptron, and who introduced it?

**Answer:** A Perceptron is the earliest and simplest type of neural network. It was invented by Frank Rosenblatt in 1958. It takes inputs, multiplies them by weights, adds a bias, and passes them through a sharp ON/OFF step function to output `0` or `1`.

### Q3: What is the purpose of the bias (b)?

**Answer:** The bias helps shift the activation function. Without a bias, the decision boundary line would always have to pass through the origin (0,0), which severely limits the patterns the neuron can learn.

### Q4: Why can a Perceptron solve an AND gate but not an XOR gate?

**Answer:** An AND gate is **linearly separable**—you can draw a single straight line on a graph to separate the `0` outputs from the `1` outputs. An XOR gate is **non-linearly separable**—the outputs are arranged diagonally, making it impossible to separate them with a single straight line.

### Q5: Can you explain visually why a single Perceptron cannot solve XOR?

**Answer:** If you plot the inputs of an XOR gate on a 2D plane:
- (0,0) and (1,1) output `0`.
- (0,1) and (1,0) output `1`.
  The zeros and ones are diagonally opposite each other. You cannot draw a single straight line that groups all the zeros on one side and all the ones on the other.

### Q6: Why do we initialize weights to zeros in Lab 1? Is zero initialization always okay?

**Answer:** In a single Perceptron, initializing weights to zero is fine because there is only one neuron. As soon as it makes a mistake, the weights will update.
However, in **deep multi-layer networks**, if we initialize all weights to zero, all neurons in a layer will learn the exact same features. Therefore, deep networks require random initialization (like Xavier or He initialization) to break symmetry.

### Q7: What is the learning rate (eta)? What happens if it is too high or too low?

**Answer:** The learning rate controls how big of a step the optimizer takes when adjusting weights.

- **Too high**: The network updates weights too aggressively, making it overshoot the best settings and fail to learn.
- **Too low**: The network updates in tiny baby steps, making it take too long to train or get stuck.

### Q8: Compare Step, Sigmoid, ReLU, and Softmax activation functions.

**Answer:**

- **Step**: A sharp ON/OFF switch. Hard to use in modern training because its derivative is zero everywhere (which blocks backpropagation).
- **Sigmoid**: A smooth curve between `0` and `1`. Great for binary output layers.
- **ReLU (Rectified Linear Unit)**: If the input is negative, it outputs `0`. If positive, it passes the value through. Highly efficient for hidden layers.
- **Softmax**: Converts a vector of scores into a multi-class probability distribution that sums up to 1.0. Used in multiclass output layers.

### Q9: Why is the Step function not used in modern backpropagation?

**Answer:** Backpropagation relies on derivatives (calculus) to adjust weights. The derivative of a step function is zero everywhere (except at 0 where it is undefined). A derivative of zero means there is no signal to tell the network how to adjust the weights, stopping learning entirely.

### Q10: What is the "vanishing gradient" problem?

**Answer:** In deep networks, gradients are multiplied backward through layers. Because the derivative of the Sigmoid function is very small (always 0.25 or less), multiplying these small values layer-by-layer causes the gradient to shrink exponentially to zero. As a result, the early layers of the network learn extremely slowly or stop training.

### Q11: Explain how we got the output of 0.5744 for input [1,1] in Lab 2.

**Answer:** 

1. We compute the score: (1 * 0.5) + (1 * 0.5) - 0.7 = 0.3.
2. We pass 0.3 into the Sigmoid formula: 1 / (1 + e^-0.3) = 0.5744.
3. Since 0.5744 is greater than or equal to 0.5, the final prediction is `1`.

### Q12: Why does Lab 3 use a hidden layer if an AND gate can be solved by a single neuron?

**Answer:** An AND gate does not require a hidden layer. However, Lab 3 uses a hidden layer to demonstrate how a Multi-Layer Perceptron (MLP) is structured and trained in Keras, serving as a stepping stone for solving more complex, non-linear problems.

### Q13: What does the code `Input(shape=(2,))` mean in Keras?

**Answer:** It defines the input layer, telling Keras that the network should expect inputs with 2 features (for example, x1 and x2 of our gate).

### Q14: How many trainable parameters are in the Keras model of Lab 3? Show the breakdown.

**Answer:** **17 parameters**:

- **Input to Hidden (4 neurons)**: (2 inputs * 4 neurons) + 4 biases = 12 parameters.
- **Hidden to Output (1 neuron)**: (4 inputs * 1 neuron) + 1 bias = 5 parameters.
- **Total**: 12 + 5 = 17.

### Q15: Why is Binary Cross-Entropy used as the loss function instead of Mean Squared Error (MSE) in Lab 3?

**Answer:** Binary Cross-Entropy penalizes wrong answers exponentially when the model is confident. If the model predicts 0.99 probability for a true label of 0, BCE gives a very high penalty. This creates steep gradients that help the model learn much faster than MSE, which gets stuck on flat parts of the Sigmoid curve.

### Q16: What is the Adam optimizer, and why is it popular?

**Answer:** Adam is an optimizer that adjusts learning rates automatically for each weight. It combines the ideas of Momentum (carrying over speed from previous steps to avoid getting stuck) and RMSProp (scaling steps based on recent gradient sizes). It is fast, stable, and requires very little manual tuning.

### Q17: Is training for 500 epochs on a dataset of 4 samples overfitting?

**Answer:** Overfitting happens when a model memorizes training data but fails on new, unseen data. In our AND gate example, the dataset of 4 samples represents the entire possible universe of inputs. Since there is no unseen data, the model cannot overfit; it is simply converging to the absolute correct answers.

### Q18: Why do we specify `dtype=np.float32` in Lab 3 inputs?

**Answer:** Deep learning frameworks like TensorFlow are optimized to perform math using 32-bit floating point numbers (`float32`). They are faster and use half the memory of Python's default 64-bit floats, which is crucial for handling large models.

### Q19: In Lab 3, why does the model output values like 0.5085 instead of exact 0s and 1s?

**Answer:** The output layer uses a Sigmoid activation, which outputs a continuous probability between `0` and `1`. It will only output exactly `0` or `1` if the weights are set to infinity. We apply a threshold (usually `0.5`) to convert these continuous probabilities into class labels.

### Q20: What is the difference between Batch, Stochastic (SGD), and Mini-Batch gradient descent?

**Answer:**

- **Batch**: Calculates gradients over the _entire_ dataset before updating weights once. Very stable, but slow for large datasets.
- **Stochastic (SGD)**: Updates weights after _every single_ data point. Fast and introduces noise that helps escape local traps, but can be erratic.
- **Mini-Batch**: Updates weights after looking at a small chunk (e.g., 32 or 64 samples) of the dataset. It balances the speed of SGD and the stability of Batch descent.

### Q21: What is Backpropagation? How does it differ from the update rule in Lab 1?

**Answer:** Backpropagation uses the calculus chain rule to calculate how much each weight in a multi-layer network contributed to the final error, propagating that error backward layer-by-layer.

- **Lab 1** does not use backpropagation because it has only a single layer. It uses the direct Perceptron learning rule, which updates weights based only on the immediate error of the output.

### Q22: What is the difference between Binary Classification and Multiclass Classification?

**Answer:**
- **Binary Classification**: Predicts one of two mutually exclusive classes (e.g., Yes/No, 0/1). The output layer typically uses 1 neuron with a **Sigmoid** activation function and **Binary Cross-Entropy** loss.
- **Multiclass Classification**: Predicts one of three or more mutually exclusive classes (e.g., Setosa / Versicolor / Virginica). The output layer uses $N$ neurons (where $N$ is the number of classes) with a **Softmax** activation function and **Categorical Cross-Entropy** loss.

### Q23: Why do we use Softmax activation instead of Sigmoid for the output layer in multiclass classification?

**Answer:** Sigmoid evaluates each output neuron independently (outputs sum to an arbitrary total). Softmax calculates probabilities relative to all classes simultaneously by exponentiating and normalizing all raw output logits. This forces the sum of probabilities across all classes to equal exactly $1.0$ ($100\%$).

### Q24: What is One-Hot Encoding, and why is it necessary for Categorical Cross-Entropy?

**Answer:** One-Hot Encoding converts integer class labels ($0, 1, 2$) into binary vectors where only the true class index is $1$ and all others are $0$ (e.g., class $2 \rightarrow [0, 0, 1]$).
It is necessary for `categorical_crossentropy` because loss is calculated by taking the dot product of the true probability vector $y$ and the log of predicted probabilities $\log(\hat{y})$. Integer labels would improperly imply a mathematical ordering or distance between classes (e.g. class 2 is twice class 1).

### Q25: Compare `categorical_crossentropy` and `sparse_categorical_crossentropy`. When should you use each?

**Answer:**
- **`categorical_crossentropy`**: Used when target labels are **One-Hot Encoded** vectors (e.g. `[[1,0,0], [0,1,0]]`).
- **`sparse_categorical_crossentropy`**: Used when target labels are **Integers** (e.g. `[0, 1, 2]`).
Both produce the exact same loss calculation and gradients; `sparse_categorical_crossentropy` simply avoids manually one-hot encoding data and saves memory.

### Q26: What is the purpose of `np.argmax()` when decoding model predictions?

**Answer:** The Softmax output layer returns a vector of probabilities (e.g., `[0.02, 0.91, 0.07]`). `np.argmax(predictions, axis=1)` finds the array index containing the maximum value (index `1` in this example), which corresponds to the predicted class integer label.

### Q27: Why is Feature Scaling (`StandardScaler`) crucial before training an MLP?

**Answer:** Neural network weight updates are directly proportional to feature magnitudes. Unscaled features with large ranges (e.g. 1000 vs 0.1) cause gradients to oscillate or blow up, leading to slow training or poor convergence. `StandardScaler` centers features to zero mean and scales them to unit variance ($z = \frac{x - \mu}{\sigma}$), ensuring smooth and balanced gradient steps across all dimensions.

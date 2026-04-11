# ML-Algorithms-From-Scratch-in-Pure-Python

Implementations of core machine learning algorithms written in pure Python — no scikit-learn, no TensorFlow, no PyTorch. Just Python and NumPy for matrix operations.

The goal is to understand what's actually happening inside these algorithms rather than treating them as black boxes.

## Algorithms Implemented

- Linear Regression (gradient descent)
- Logistic Regression
- K-Nearest Neighbors (KNN)

> Each algorithm is implemented in its own file with clear, readable code.

## Why From Scratch?

Using a library like scikit-learn is one line of code. Writing the algorithm yourself means you understand:
- How gradient descent actually updates weights
- How a decision tree splits on features
- How KNN computes distances and votes

This repo is the proof of that understanding.

## Structure

```
ML-Algorithms-From-Scratch-in-Pure-Python/
├── linear_regression.py
├── logistic_regression.py
├── knn.py
```

## Usage

Each file is self-contained. Run any file directly:

```bash
python linear_regression.py
python knn.py
```

## Stack

- Python 3
- NumPy (math only — no ML abstractions)

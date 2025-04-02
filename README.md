# Light Gradient Boosting Machine (LightGBM)

This repository contains various experiments and implementations using **LightGBM**, a gradient boosting framework that uses tree-based learning algorithms. LightGBM is designed for efficiency and high performance and is widely used for classification and regression tasks.

## About LightGBM

LightGBM uses two main techniques to improve efficiency:

1. **Gradient-based One-Side Sampling (GOSS)**: Excludes a significant portion of data with small gradients, keeping only the relevant samples for better information gain.
2. **Exclusive Feature Bundling (EFB)**: Reduces memory usage by bundling mutually exclusive features, improving computational efficiency.

### Key Features

- Faster training speed and high efficiency
- Lower memory usage
- Handles large datasets effectively
- Better accuracy compared to other boosting algorithms
- Supports parallel and GPU learning
- Handles overfitting well with small datasets

---

## 📂 Repository Contents

### 1️⃣ **Regression using LightGBM**

- Implements `LGBMRegressor` on a generated dataset.
- Example usage:
  ```python
  from lightgbm import LGBMRegressor
  model = LGBMRegressor()
  model.fit(X, y)
  pred = model.predict([test_sample])

  
  ### 2️⃣ **Effect of Boosting Type on Performance**

- Compares different boosting types (`gbdt`, `dart`, `goss`).
- Uses cross-validation to evaluate model performance.
- Example results:
  ``` python
  gbdt 0.925 (0.031) dart 0.912 (0.028) goss 0.918 (0.027)

  
### 3️⃣ **Effect of Number of Trees on Performance**

- Evaluates model performance for different numbers of trees (`n_estimators`).
- Uses **box plots** to compare results.
- Example models tested: `[10, 50, 100, 500, 1000, 5000]`.

---

##  Installation

To install the required dependencies, run:
```bash
   pip install lightgbm numpy scikit-learn matplotlib
```

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/lightGBM.git
   cd lightGBM
2. Run the Jupyter Notebook:

   ```bash
   jupyter notebook lightGBM.ipynb

## Results & Observations

- GOSS performs better for smaller datasets due to its selective data usage.

- GBDT provides overall balanced accuracy.

- Increasing trees improves accuracy, but after a point, it leads to overfitting.

##  References

- [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)
- [Gradient Boosting Explained](https://en.wikipedia.org/wiki/Gradient_boosting)




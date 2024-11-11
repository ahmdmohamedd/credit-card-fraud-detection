# Credit Card Fraud Detection using Logistic Regression

This repository contains a MATLAB-based implementation of a **Credit Card Fraud Detection** system using **Logistic Regression**. The model is trained on a balanced dataset with **hybrid sampling**, combining both **oversampling** of the minority class (fraud) and **undersampling** of the majority class (non-fraud). The system is designed to predict fraudulent transactions with a high level of accuracy and reliability.

## Features
- **Logistic Regression** model for binary classification (Fraud vs. Non-Fraud).
- **Hybrid Sampling** technique to balance the dataset and improve performance.
- **Threshold tuning** at 0.6 for optimized **precision** and **recall**.
- Visualizations of:
  - **Fraud vs Non-Fraud Transactions**.
  - **Confusion Matrix** for model evaluation.

## Files in this Repository:
1. **`credit_card_fraud_detection.m`**: MATLAB script that performs the following tasks:
   - Data loading and preprocessing (including feature scaling).
   - Hybrid sampling (oversampling of fraud and undersampling of non-fraud cases).
   - Logistic regression model training with gradient descent.
   - Model evaluation using precision, recall, F1-score, and confusion matrix.
   - Visualization of key results like fraud detection and confusion matrix.

2. **`Fraud vs Non-Fraud Transactions.png`**: A scatter plot showing the distribution of fraud vs non-fraud transactions after hybrid sampling.

3. **`confusion_matrix.png`**: Visualization of the confusion matrix generated during model evaluation, showing the true positives, false positives, true negatives, and false negatives.

## Installation

### Requirements:
- MATLAB (version 2019 or later recommended)
- `readtable` function (MATLAB built-in)
- No additional toolboxes required

### Steps to Run:
1. Download the repository or clone it to your local machine:
   ```
   git clone https://github.com/ahmdmohamedd/credit-card-fraud-detection.git
   ```
   
2. Ensure the **`creditcard.csv`** dataset is available and loaded in the script.

3. Open the `credit_card_fraud_detection.m` script in MATLAB.

4. Run the script in MATLAB. The output will display the model's performance metrics (accuracy, precision, recall, F1-score) and plot the confusion matrix and fraud vs non-fraud transactions.

## Evaluation Metrics

- **Accuracy**: 90.59% — The percentage of total correct predictions.
- **Precision**: 0.85 — The percentage of predicted fraud cases that were actually fraud.
- **Recall**: 0.98 — The percentage of actual fraud cases identified by the model.
- **F1-Score**: 0.91 — The harmonic mean of precision and recall.

## Results
The model performs well in detecting fraud with high recall (0.98), ensuring that almost all fraudulent transactions are identified. The precision (0.85) is also strong, indicating that most of the predicted fraud cases are accurate.

### Confusion Matrix:
```
    Predicted Fraud | Predicted Non-Fraud
    --------------------------------------
    Actual Fraud    | 139,838         | 23,983
    Actual Non-Fraud| 2,811           | 118,174
```

## Contributing
Contributions are welcome! If you have suggestions for improvements, feel free to fork the repository and create a pull request.

```

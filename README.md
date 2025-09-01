## Project Overview
This project was developed for the **2025 SinoPac AI-GO competition**, where the objective is to build a machine learning model to classify whether a stock will become a **momentum stock (y=1)** or **non-momentum stock (y=0)**.  

The problem is formulated as a **binary classification task**, and we focus on feature engineering, machine learning models, and evaluation metrics to maximize predictive performance.

---

## Methods

### 1. Data Preprocessing
- Handled missing values using imputation strategies.  
- Normalized numerical features to ensure scale consistency.  
- Encoded categorical features (if any).  
- Split the dataset into **training / validation / testing** sets.  

### 2. Feature Engineering
We extracted features from both market and financial data:
- **Technical indicators**: moving average (MA), relative strength index (RSI), moving average convergence divergence (MACD).  
- **Financial ratios**: earnings per share (EPS), return on equity (ROE).  
- **Temporal patterns**: rolling statistics of price and volume across past N days.  

### 3. Models
We experimented with both **traditional machine learning** and **deep learning** approaches:  

- **Machine Learning**:  
  - Logistic Regression (baseline)  
  - Random Forest  
  - XGBoost  
  - LightGBM  

- **Deep Learning**:  
  - Multilayer Perceptron (MLP) using PyTorch  
  - Recurrent models (LSTM) to capture sequential dependencies  

### 4. Model Training & Optimization
- Hyperparameter tuning with **cross-validation**.  
- Early stopping to avoid overfitting.  
- Feature importance analysis for interpretability (tree-based models).  

### 5. Evaluation Metrics
We evaluated models using multiple metrics to ensure robustness:
- **Accuracy**  
- **F1-score**  
- **ROC-AUC**  


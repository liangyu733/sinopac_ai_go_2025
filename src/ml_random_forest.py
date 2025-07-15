
# %% import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import predicted_evaluation
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% import dataset
dat = pd.read_csv('../data/col707.csv')
X_test = dat.drop(columns=['Unnamed: 0', 'ID', '飆股'])
y_test = dat['飆股'].values
X_0 = X_test[y_test == 0].dropna()

# %% impute  missing values for y = 1
X_1 = X_test[y_test == 1]
imputed_values = X_1.median()
X_1 = X_1.fillna(imputed_values)

# %% candidate population for training
X_cand = pd.concat([X_0, X_1])
y_cand = y_test[X_cand.index]

# %% data transformation
scaler = StandardScaler()
X_cand = pd.DataFrame(scaler.fit_transform(X_cand), 
                      columns = X_cand.columns, 
                      index = X_cand.index)
# %% oversampling to balance classes
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_augcand, y_augcand = smote.fit_resample(X_cand, y_cand)

# %% 
dcand = xgb.DMatrix(X_augcand, label = y_augcand)
dtest = xgb.DMatrix(X_test, label = y_test)

# %% random forest model
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(
    n_estimators = 100, 
    max_depth = 10,
    n_jobs = 20,
    min_samples_leaf = 10,       
    max_features = 'sqrt',        
    class_weight = 'balanced'   
    )

rf_clf.fit(X_augcand, y_augcand)

# %%
y_prob_rf = rf_clf.predict_proba(X_augcand)[:, 1]
y_pred_rf = rf_clf.predict(X_augcand)

from predicted_evaluation import
results = evaluate_model(y_augcand, y_prob_rf, threshold = 0.5)
# %% feature selection by importance (top 30)
importances = rf_clf.feature_importances_
features = X_augcand.columns

importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(importance_df['feature'][:30], importance_df['importance'][:30]) 
plt.xticks(rotation=90)
plt.ylabel("Feature Importance")
plt.title("Top 30 Feature Importances from Random Forest")
plt.tight_layout()
plt.show()
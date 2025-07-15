# %% Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from predicted_evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from IPython.display import display

# %%
dat = pd.read_csv("../data/col707.csv")

# %%
dat.drop(columns = ['Unnamed: 0', 'ID'], inplace=True)

# %% check NAs
dat.isna().sum()

# %% filter data with no NAs
na_cnt = dat.isna().sum(axis = 1)
print(na_cnt)
idx_y1 = dat[(na_cnt == 0) & (dat['飆股'] == 1)].index
idx_y0 = dat[(na_cnt == 0) & (dat['飆股'] == 0)].index
n0 = len(idx_y0)
n1 = len(idx_y1)

# %% create data for analysis
idx_train = idx_y0.union(idx_y1)
y = dat.loc[idx_train, '飆股']
X = dat.loc[idx_train, dat.columns != '飆股']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# %% weights for GLM
wt = y.map({0: n0/(n0+n1), 1: n1/(n0+n1)})

# %% LASSO
lasso_clf = LogisticRegression(
    penalty='l1',            # Lasso（L1）
    solver='liblinear',      # solver for L1 penalty
    max_iter=1000,           # maximum iterations
    C = 1.0,                 # strength of regularization (1/lambda)
    class_weight='balanced'  # deal with class imbalance
)
# %% weights for LASSO
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
print(dict(zip(classes, weights)))

# %% model fitting of LASSO
lasso_clf.fit(X_scaled, y)
# %% coefficients of the model
coef = lasso_clf.coef_.ravel()  # to flatten the array
coef_df = pd.Series(coef, index=X.columns)
nonzero_coef = coef_df[coef_df != 0].sort_values(ascending=False)
print(nonzero_coef)

# %% filter features by LASSO
X_selected = X_scaled[nonzero_coef.index]
print(X_selected.columns)

# %% prediction of the model
yprob_lasso = lasso_clf.predict_proba(X_scaled)[:, 1] # threshold = 0.5
yhat_lasso = lasso_clf.predict(X_scaled)
result_lasso = pd.DataFrame({
    'y_true': y,
    'y_prob': yprob_lasso,
    'y_pred': yhat_lasso
}, index=y.index)
print(result_lasso.head())

# %%
display(result_lasso)

# %% evaluation of Lasso
evaluate_lasso = evaluate_model(y, yhat_lasso)
print(evaluate_lasso["conf_matrix"])
print(evaluate_lasso["metrics"])
print("AUC:", evaluate_lasso["auc"])

# %% computation of VIF
def vif(X: pd.DataFrame) -> pd.DataFrame:
    # Z-transformation
    X_scaled = (X - X.mean()) / X.std(ddof=0)
    
    # Correlation matrix
    corr_matrix = np.corrcoef(X_scaled.T)

    # Inverse of correlation matrix
    inv_corr = np.linalg.inv(corr_matrix)

    # VIF = corresponding diagonal elements
    vif_values = np.diag(inv_corr)

    return pd.DataFrame({
        "feature": X.columns,
        "VIF": vif_values
    })

# %% compute VIF for selected features
vif_df = vif(X_selected)
print(vif_df)

# %% plot VIF
vif_threshold = 10
vif_df["log10_VIF"] = np.log10(vif_df["VIF"])
vif_df_sorted = vif_df.sort_values(by="log10_VIF", ascending=False)
log10_threshold = np.log10(vif_threshold)
n_total = len(vif_df_sorted)
n_removed = (vif_df_sorted["VIF"] > vif_threshold).sum()
n_retained = n_total - n_removed

plt.figure(figsize=(12, 6))
plt.bar(vif_df_sorted["feature"], vif_df_sorted["log10_VIF"], label="Features")
plt.xticks([], [])
plt.axhline(y=log10_threshold, color='red', linestyle='--', label="VIF = 10")
plt.ylabel("log10(VIF)")
plt.title(f"VIF by Feature (Removed: {n_removed}, Retained: {n_retained})")
plt.legend()
plt.tight_layout()
plt.show()

# %% filter features by VIF
filter_index = vif_df[vif_df["VIF"] <= vif_threshold]["feature"].tolist()
X_filtered = X_selected[filter_index]

# %% model matrix for GLM
X_filtered = sm.add_constant(X_filtered, has_constant='add')  # 防止重複加
# %% GLM model fitting
model_glm = sm.GLM(y, X_filtered, family=sm.families.Binomial(), freq_weights=wt)
result_glm = model_glm.fit()
print(result_glm.summary())

# %% predicted probability of GLM
yprob_glm = result_glm.predict(X_filtered)

# %% evaluation of GLM
evaluate_glm = evaluate_model(y, yprob_glm)
print(evaluate_glm["conf_matrix"])
print(evaluate_glm["metrics"])
print("AUC:", evaluate_glm["auc"])

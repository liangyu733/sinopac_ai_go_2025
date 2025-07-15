# %% 
params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',  # Use GPU for faster training
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc'
}
bst = xgb.train(params, dcand, num_boost_round = 100)

y_prob = bst.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)
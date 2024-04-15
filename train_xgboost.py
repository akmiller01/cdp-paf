import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score

crs = pd.read_csv('large_data/meta_model_for_xgboost.csv')
crs.pop('limited_text')
dataset = pd.read_csv('large_data/predicted_meta_model_data_combo.csv')
# dataset = dataset[dataset["Crisis finance actual"] == True]
dataset = pd.merge(dataset, crs, on="full_text")
pre_dum_cols = dataset.columns.values.tolist()

dataset = pd.get_dummies(dataset, columns=["DonorName", "SectorName", "PurposeName", "FlowName"], dtype=float)
post_dum_cols = dataset.columns.values.tolist()

dummy_cols = [col for col in post_dum_cols if col not in pre_dum_cols]
cf_X_cols = ['Crisis finance confidence'] + dummy_cols
paf_X_cols = ["PAF confidence"] + dummy_cols
aa_X_cols = ["AA confidence"] + dummy_cols
cf_X = dataset[cf_X_cols].values.astype(float)
paf_X = dataset[paf_X_cols].values.astype(float)
aa_X = dataset[aa_X_cols].values.astype(float)

cf_y = dataset.pop('Crisis finance actual').values.astype(float)
paf_y = dataset.pop('PAF actual').values.astype(float)
aa_y = dataset.pop('AA actual').values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(cf_X, cf_y, test_size=0.2, shuffle=True)

xgb_reg = xgb.XGBClassifier(n_estimators=100)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
print("\nCF XGBoost:")
print("MSE: ", mse_xgb)
print("Precision: ", prec_xgb)
print("Recall: ", rec_xgb)

X_train, X_test, y_train, y_test = train_test_split(paf_X, paf_y, test_size=0.2, shuffle=True)

xgb_reg = xgb.XGBClassifier(n_estimators=100)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
print("\nPAF XGBoost:")
print("MSE: ", mse_xgb)
print("Precision: ", prec_xgb)
print("Recall: ", rec_xgb)

X_train, X_test, y_train, y_test = train_test_split(aa_X, aa_y, test_size=0.2, shuffle=True)

xgb_reg = xgb.XGBClassifier(n_estimators=100)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
print("\nAA XGBoost:")
print("MSE: ", mse_xgb)
print("Precision: ", prec_xgb)
print("Recall: ", rec_xgb)

cf_xgb = xgb.XGBClassifier(n_estimators=100)
cf_xgb.fit(cf_X, cf_y)
paf_xgb = xgb.XGBClassifier(n_estimators=100)
paf_xgb.fit(paf_X, paf_y)
aa_xgb = xgb.XGBClassifier(n_estimators=100)
aa_xgb.fit(aa_X, aa_y)


with open('models/xgb.pkl', 'wb') as f:
    pickle.dump((
        cf_xgb, paf_xgb, aa_xgb, dummy_cols
    ), f)
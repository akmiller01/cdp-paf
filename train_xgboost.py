from sklearn.metrics import confusion_matrix
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
    }

confusion_indices = ["Actual negative", "Actual positive"]
confusion_columns = ["Predicted negative", "Predicted positive"]

crs = pd.read_csv('large_data/meta_model_for_xgboost.csv')
crs.pop('limited_text')
crs = crs.drop_duplicates()
dataset = pd.read_csv('large_data/predicted_meta_model_data_combo.csv')
# dataset = dataset[dataset["Crisis finance actual"] == True]
dataset = pd.merge(dataset, crs, on="full_text", how="left")

print("\nCF:")
print("Accuracy: ", accuracy_score(dataset['Crisis finance actual'], dataset['Crisis finance predicted']))
print("Recall: ", recall_score(dataset['Crisis finance actual'], dataset['Crisis finance predicted']))
print("Precision: ", precision_score(dataset['Crisis finance actual'], dataset['Crisis finance predicted']))

print("\nPAF:")
print("Accuracy: ", accuracy_score(dataset['PAF actual'], dataset['PAF predicted']))
print("Recall: ", recall_score(dataset['PAF actual'], dataset['PAF predicted']))
print("Precision: ", precision_score(dataset['PAF actual'], dataset['PAF predicted']))

print("\nAA:")
print("Accuracy: ", accuracy_score(dataset['AA actual'], dataset['AA predicted']))
print("Recall: ", recall_score(dataset['AA actual'], dataset['AA predicted']))
print("Precision: ", precision_score(dataset['AA actual'], dataset['AA predicted']))

pre_dum_cols = dataset.columns.values.tolist()

dataset = pd.get_dummies(dataset, columns=["DonorName", "SectorName", "PurposeName", "FlowName"], dtype=float)
post_dum_cols = dataset.columns.values.tolist()

dummy_cols = [col for col in post_dum_cols if col not in pre_dum_cols]
cf_X_cols = ['Crisis finance confidence'] + dummy_cols
paf_X_cols = ['Crisis finance confidence', "PAF confidence"] + dummy_cols
aa_X_cols = ['Crisis finance confidence', "PAF confidence", "AA confidence"] + dummy_cols
cf_X = dataset[cf_X_cols].values.astype(float)
paf_X = dataset[paf_X_cols].values.astype(float)
aa_X = dataset[aa_X_cols].values.astype(float)

cf_y = dataset.pop('Crisis finance actual').values.astype(float)
paf_y = dataset.pop('PAF actual').values.astype(float)
aa_y = dataset.pop('AA actual').values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(cf_X, cf_y, test_size=0.5, shuffle=True, random_state=0)

xgb_reg = xgb.XGBClassifier(n_estimators=100)
random_search = RandomizedSearchCV(xgb_reg, param_distributions=params, n_iter=5, scoring='f1', n_jobs=4, cv=2, random_state=1001)
random_search.fit(X_train, y_train)
y_pred_xgb = random_search.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
print("\nCF XGBoost:")
print("Accuracy: ", acc_xgb)
print("Recall: ", rec_xgb)
print("Precision: ", prec_xgb)
mcm = confusion_matrix(y_test, y_pred_xgb)
crisis_confusion = pd.DataFrame(mcm, index=confusion_indices, columns=confusion_columns)
crisis_confusion.to_csv('data/crisis_confusion_xgb.csv')
print(crisis_confusion)
cf_params = random_search.best_params_
print(random_search.best_params_)

X_train, X_test, y_train, y_test = train_test_split(paf_X, paf_y, test_size=0.5, shuffle=True, random_state=0)

xgb_reg = xgb.XGBClassifier(n_estimators=100)
random_search = RandomizedSearchCV(xgb_reg, param_distributions=params, n_iter=5, scoring='f1', n_jobs=4, cv=2, random_state=1001)
random_search.fit(X_train, y_train)
y_pred_xgb = random_search.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
print("\nPAF XGBoost:")
print("Accuracy: ", acc_xgb)
print("Recall: ", rec_xgb)
print("Precision: ", prec_xgb)
mcm = confusion_matrix(y_test, y_pred_xgb)
paf_confusion = pd.DataFrame(mcm, index=confusion_indices, columns=confusion_columns)
paf_confusion.to_csv('data/paf_confusion_xgb.csv')
print(paf_confusion)
print(random_search.best_params_)
paf_params = random_search.best_params_

X_train, X_test, y_train, y_test = train_test_split(aa_X, aa_y, test_size=0.5, shuffle=True, random_state=0)

xgb_reg = xgb.XGBClassifier(n_estimators=100)
random_search = RandomizedSearchCV(xgb_reg, param_distributions=params, n_iter=5, scoring='f1', n_jobs=4, cv=2, random_state=1001)
random_search.fit(X_train, y_train)
y_pred_xgb = random_search.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
print("\nAA XGBoost:")
print("Accuracy: ", acc_xgb)
print("Recall: ", rec_xgb)
print("Precision: ", prec_xgb)
mcm = confusion_matrix(y_test, y_pred_xgb)
aa_confusion = pd.DataFrame(mcm, index=confusion_indices, columns=confusion_columns)
aa_confusion.to_csv('data/aa_confusion_xgb.csv')
print(aa_confusion)
print(random_search.best_params_)
aa_params = random_search.best_params_

cf_xgb = xgb.XGBClassifier(n_estimators=100, **cf_params)
cf_xgb.fit(cf_X, cf_y)
paf_xgb = xgb.XGBClassifier(n_estimators=100, **paf_params)
paf_xgb.fit(paf_X, paf_y)
aa_xgb = xgb.XGBClassifier(n_estimators=100, **aa_params)
aa_xgb.fit(aa_X, aa_y)


with open('models/xgb.pkl', 'wb') as f:
    pickle.dump((
        cf_xgb, paf_xgb, aa_xgb, dummy_cols
    ), f)
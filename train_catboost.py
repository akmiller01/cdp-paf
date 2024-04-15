from sklearn.metrics import confusion_matrix
import pandas as pd
from catboost import Pool, CatBoostClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
print("Precision: ", precision_score(dataset['Crisis finance actual'], dataset['Crisis finance predicted']))
print("Recall: ", recall_score(dataset['Crisis finance actual'], dataset['Crisis finance predicted']))

print("\nPAF:")
print("Accuracy: ", accuracy_score(dataset['PAF actual'], dataset['PAF predicted']))
print("Precision: ", precision_score(dataset['PAF actual'], dataset['PAF predicted']))
print("Recall: ", recall_score(dataset['PAF actual'], dataset['PAF predicted']))

print("\nAA:")
print("Accuracy: ", accuracy_score(dataset['AA actual'], dataset['AA predicted']))
print("Precision: ", precision_score(dataset['AA actual'], dataset['AA predicted']))
print("Recall: ", recall_score(dataset['AA actual'], dataset['AA predicted']))


dummy_cols = ["DonorName", "SectorName", "PurposeName", "FlowName"]
cf_X_cols = ['Crisis finance confidence', ] + dummy_cols
paf_X_cols = ["PAF confidence"] + dummy_cols
aa_X_cols = ["AA confidence"] + dummy_cols
cf_X = dataset[cf_X_cols].values
paf_X = dataset[paf_X_cols].values
aa_X = dataset[aa_X_cols].values

cf_y = dataset.pop('Crisis finance actual').values.astype(float)
paf_y = dataset.pop('PAF actual').values.astype(float)
aa_y = dataset.pop('AA actual').values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(cf_X, cf_y, test_size=0.5, shuffle=True, random_state=0)

# initialize Pool
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=[1, 2, 3, 4])
test_pool = Pool(X_test, 
                 cat_features=[1, 2, 3, 4]) 

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=False)
# train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_pred_cat = model.predict(test_pool)
y_prob_cat = model.predict_proba(test_pool)

acc_cat = accuracy_score(y_test, y_pred_cat)
prec_cat = precision_score(y_test, y_pred_cat)
rec_cat = recall_score(y_test, y_pred_cat)
print("\nCF catboost:")
print("Accuracy: ", acc_cat)
print("Precision: ", prec_cat)
print("Recall: ", rec_cat)
mcm = confusion_matrix(y_test, y_pred_cat)
crisis_confusion = pd.DataFrame(mcm, index=confusion_indices, columns=confusion_columns)
crisis_confusion.to_csv('data/crisis_confusion_cat.csv')
print(crisis_confusion)

X_train, X_test, y_train, y_test = train_test_split(paf_X, paf_y, test_size=0.5, shuffle=True, random_state=0)

# initialize Pool
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=[1, 2, 3, 4])
test_pool = Pool(X_test, 
                 cat_features=[1, 2, 3, 4]) 

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=False)
# train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_pred_cat = model.predict(test_pool)
y_prob_cat = model.predict_proba(test_pool)

acc_cat = accuracy_score(y_test, y_pred_cat)
prec_cat = precision_score(y_test, y_pred_cat)
rec_cat = recall_score(y_test, y_pred_cat)
print("\nPAF catboost:")
print("Accuracy: ", acc_cat)
print("Precision: ", prec_cat)
print("Recall: ", rec_cat)
mcm = confusion_matrix(y_test, y_pred_cat)
paf_confusion = pd.DataFrame(mcm, index=confusion_indices, columns=confusion_columns)
paf_confusion.to_csv('data/paf_confusion_cat.csv')
print(paf_confusion)

X_train, X_test, y_train, y_test = train_test_split(aa_X, aa_y, test_size=0.5, shuffle=True, random_state=0)

# initialize Pool
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=[1, 2, 3, 4])
test_pool = Pool(X_test, 
                 cat_features=[1, 2, 3, 4]) 

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=False)
# train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_pred_cat = model.predict(test_pool)
y_prob_cat = model.predict_proba(test_pool)

acc_cat = accuracy_score(y_test, y_pred_cat)
prec_cat = precision_score(y_test, y_pred_cat)
rec_cat = recall_score(y_test, y_pred_cat)
print("\nAA catboost:")
print("Accuracy: ", acc_cat)
print("Precision: ", prec_cat)
print("Recall: ", rec_cat)
mcm = confusion_matrix(y_test, y_pred_cat)
aa_confusion = pd.DataFrame(mcm, index=confusion_indices, columns=confusion_columns)
aa_confusion.to_csv('data/aa_confusion_cat.csv')
print(aa_confusion)
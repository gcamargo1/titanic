# plot roc curve

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import Imputer

# Load dataset
df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# Preprocess
le = preprocessing.LabelEncoder()
df['sex_val'] = le.fit_transform(df['Sex'])
df_test['sex_val'] = le.fit_transform(df_test['Sex'])

# imputation
df['age_vals'] = df.Age.fillna(int(df.Age.mean()))
df_test['age_vals'] = df_test.Age.fillna(int(df_test.Age.mean()))

# Select variables
y = df[['Survived']]
predictors = ['Pclass', 'sex_val', 'age_vals']
X = df[predictors]
X_test_test = df_test[predictors]

# how to convert class to
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X_train, y_train.values.ravel())

# prediction
y_pred = clf.predict(X_test)
y_score = pd.DataFrame(clf.predict_proba(X_test))[1]
# accuracy
acc = metrics.accuracy_score(y_test, y_pred)
# auc
auc = metrics.roc_auc_score(y_test, y_score)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

print auc

y_pred_test = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Survived':clf.predict(X_test_test)})
y_pred_test.to_csv('./submissions/submission3.csv', index=False)
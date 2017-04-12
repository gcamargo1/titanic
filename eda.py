"""Guidance from https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/
https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
"""
from sklearn.preprocessing import Imputer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
sns.set(style="white", color_codes=True)
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
import statsmodels.formula.api as smf


# 0) Load datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Variable identification: predictors(input) and target(output)
target = 'Survived'
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare']

# identify the data type (char, numeric) and category (categorical/continuous) of the variables.
cols = train.columns
types = []
for col in cols:
    type_ = train[col].dtype.char
    types.append([col, type_])

# Univariate analysis
# continuous variable
continuous_vars = ['Age', 'Fare']
for var in continuous_vars:
    cont_var = train[var]
    # Central tendency
    mean = cont_var.mean()
    median = cont_var.median()
    mode = cont_var.mode()
    mn = cont_var.min()
    mx = cont_var.min()
    # measure of dispersion
    rg = mx - mn
    std = cont_var.std()
    var = cont_var.var()
    #skew = cont_var.skew
    #kurtosis = cont_var.kurtosis
    # Plots
    hist = cont_var.hist()
    boxplot = cont_var.plot(kind='box')

# categorical variables
cat_vars = ['Pclass', 'Sex', 'SibSp', 'Parch']

for var in cat_vars:
    cat_var = train[var]
    freq_table = cat_var.value_counts()
    bar_plot = freq_table.plot(kind='bar')

# Bi-variate Analysis

# Continuous & Continuous
t = train[continuous_vars]
corr = t.corr()
# Categorical & Categorical
t = train[cat_vars]
for i in range(len(cat_vars)):
    if i < len(cat_vars)-1:
        tab = pd.crosstab(t.iloc[:, i], t.iloc[:, i+1])
        tab.plot(kind='bar', stacked=True)
    else:
        tab = pd.crosstab(t.iloc[:, i], t.iloc[:, 0])
        tab.plot(kind='bar', stacked=True)
    # Chi-square test of variable independence - http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-25-chi.html
    indep_p = stats.chi2_contingency(observed=tab)[1]

# Categorical & Continuous
results = []
for cont in continuous_vars:
    for cat in cat_vars:
        data = train[[cont, cat]]
        data.boxplot(column=cont, by=cat)
        formula = "{} ~ {}".format(cont, cat)
        mod = smf.ols(formula=formula, data=data).fit()
        pvalue = mod.pvalues[1]
        results.append([cont, cat, pvalue])
# convert categorical to integer
# imputation
# find missing values
for predictor in continuous_vars:
    df = train[predictor]
    if df.dtype.char == 'O':
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        imp.fit(df.values.reshape(-1, 1))
        train[predictor + '_imp'] = imp.transform(df.values.reshape(-1, 1))
    if df.dtype == 'd':

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df.values.reshape(-1, 1))
        train[predictor + '_imp'] = imp.transform(df.values.reshape(-1, 1))

# 1) Inspect the distribution of target variable.
# Check response balance
#train.Survived.value_counts()
#sns.FacetGrid(train, hue="Survived", size=5) \
#   .map(plt.scatter, "Sex", "Age") \
#   .add_legend()
#sns.boxplot(x="Survived", y="Age", data=train)

#2) Data Preprocessing

# Sometimes several files are provided and we need to join them.

# Deal with missing data.
train['age'] = train.Age.fillna(int(train.Age.mean()))
test['age'] = test.Age.fillna(int(test.Age.mean()))

# Deal with outliers.

# Encode categorical variables if necessary.
le = preprocessing.LabelEncoder()
train['sex'] = le.fit_transform(train['Sex'])
test['sex'] = le.fit_transform(test['Sex'])
le2 = preprocessing.LabelEncoder()
train['cabin'] = le2.fit_transform(train['Cabin'])
test['cabin'] = le2.fit_transform(test['Cabin'])
le3 = preprocessing.LabelEncoder()
train['embarked'] = le3.fit_transform(train['Embarked'])
test['embarked'] = le3.fit_transform(test['Embarked'])

le4 = preprocessing.LabelEncoder()
train['fare'] = le3.fit_transform(train['Fare'])
test['fare'] = le3.fit_transform(test['Fare'])

# Feature Engineering
# Feature Selection
y = train[['Survived']]
predictors = ['Pclass', 'sex', 'age', 'embarked', 'cabin', 'fare', 'SibSp',
              'Parch']

X = train[predictors]
X_test = test[predictors]
clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X, y.values.ravel())

y_pred_test = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':clf.predict(X_test)})
y_pred_test.to_csv('./submissions/submission7.csv', index=False)
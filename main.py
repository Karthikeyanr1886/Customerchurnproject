import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('customer_churn.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
df.dropna(subset=['TotalCharges'],inplace=True)
senior_male_electronic =df [(df['gender'] == 'Male')   & (df['SeniorCitizen'] == 1) & (df['PaymentMethod'] =='Electronic check')]
tenure_or_charges = df[
  (df["tenure"] > 70) | (df["MonthlyCharges"] > 100)
]
sns.set_theme(style="whitegrid")
sns.countplot(
  data=df,
  x="InternetService",
  order=df["InternetService"].value_counts().index,
)
plt.title("Customer Trends")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.show()
df = df.drop('customerID', axis = 1)
from sklearn.preprocessing import LabelEncoder
cat_col = df.select_dtypes(include = 'object').columns
le = LabelEncoder()
for col in cat_col:
  df[col]  = le.fit_transform(df[col])
  from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
x = df.drop('Churn', axis =1)
y = df['Churn']
x_train , x_test, y_train, y_test = train_test_split(x, y, train_size = 0.80,random_state = 0)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
y_pred = logmodel.predict(x_test)
print("accuracy score", accuracy_score(y_test, y_pred))
print("\n confusion matrix \n", confusion_matrix(y_test, y_pred))
from sklearn.tree  import DecisionTreeClassifier
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.80, random_state  = 42)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
y_pred1 = dt_model.predict(X_test)
print("decision tree acc ", accuracy_score(Y_test, y_pred1))
print("confusion mat \n", confusion_matrix(Y_test, y_pred1))
from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(X_train, Y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random forest", accuracy_score(Y_test, y_pred_rf))
print("confusion matrix \n", confusion_matrix(Y_test , y_pred_rf))
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators' : [50,100],
    'max_depth' : [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state = 0)
grid = GridSearchCV(
    estimator = rf,
    param_grid =param_grid,
    cv = 5
)
grid.fit(x_train, y_train)
best_rf = grid.best_estimator_
grid.best_params_
y_pred1= best_rf.predict(x_test)
print("accuracy score", accuracy_score(y_test, y_pred1))

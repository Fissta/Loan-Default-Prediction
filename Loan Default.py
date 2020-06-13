import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBRegressor
import scikitplot as skplt
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('datasets_906_1654_loan_data.csv')
df.info
df.describe()
df.head()
plt.figure()
df[df['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30)
label = 'Credit Policy = 1'
df[df['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30)
label = 'Credit Policy = 0'
plt.legend()
plt.xlabel('FICO')
plt.show()

plt.figure()
df[df['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30)
label = 'Not Fully paid = 1'
df[df['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30)
label = 'Not fully paid = 1'
plt.legend()
plt.xlabel('fico')
plt.show()

cat_features = ['purpose']
final_data = pd.get_dummies(df, columns=cat_features, drop_first=True)
final_data.info()
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

dt_model = DecisionTreeClassifier(max_leaf_nodes=250, random_state=1)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mae = mean_absolute_error(dt_predictions, y_test)

print(f'Mean Absolute Error for Decision Tree is {dt_mae}\n')
print(classification_report(y_test, dt_predictions))

print(confusion_matrix(y_test, dt_predictions))
skplt.metrics.plot_confusion_matrix(y_test, dt_predictions, figsize=(10, 8))

rf_model = RandomForestClassifier(n_estimators=250, random_state=1)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_mae = mean_absolute_error(rf_predict, y_test)
print(f'Mean Absolute Error for Random Forest Tree is {rf_mae}\n')
print(classification_report(y_test, rf_predict))

print(confusion_matrix(y_test, rf_predict))
skplt.metrics.plot_confusion_matrix(y_test, rf_predict, figsize=(10, 8))

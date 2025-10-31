
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

file_path = 'order.csv'

df = pd.read_csv(file_path)

numerical_df = df.select_dtypes(include = 'number')
categorical_df = df.select_dtypes(include = 'object')
top_15_items = df['Items in order'].value_counts().nlargest(15).index
df_top_15 = df[df['Items in order'].isin(top_15_items)]



#Data visualization (remove triple quotes and run. Note : it might take a while to run)
'''for col in numerical_df.columns :
  plt.figure(figsize = (10, 8))
  sns.histplot(data = df, x = col, kde = True)


for col in categorical_df.columns :
  if col in ['Items in order', 'Review']:
    continue
  plt.figure(figsize = (10, 8))
  sns.countplot(data = df, x = col)


plt.figure(figsize = (10, 8))
sns.countplot(data = df_top_15, y = 'Items in order', order = df_top_15)
sns.countplot(data = df_top_15, y = 'Items in order')


for col in numerical_df.columns :
 plt.figure(figsize = (10, 8))
 sns.boxplot(data = df_top_15, x = col, y = 'Items in order', order = top_15_items)


for col in categorical_df.columns :
 if col in ['Items in order', 'Review'] :
   continue
 ct = pd.crosstab(df_top_15[col], df_top_15['Items in order'])
 plt.figure(figsize = (10, 8))
 sns.heatmap(ct, annot = True, fmt = 'd', cmap = 'viridis')


plt.show()

'''

cols_to_drop = [
    'Order ID',
    'Customer ID',
    'Instructions',
    'Discount construct',

    'City',
    'Delivery',
    'Gold discount',
    'Brand pack discount',
    'Restaurant penalty (Rejection)',
    'Restaurant discount (Promo)',
    'Restaurant discount (Flat offs, Freebies & others)',
    'Review',

    'Order Status',
    'Cancellation / Rejection reason',
    'Order Ready Marked',
    'Customer complaint tag',
    'Rating',
    'Restaurant compensation (Cancellation)',
    'Bill subtotal',
    'Packaging charges',
    'Total',
    'KPT duration (minutes)',
    'Rider wait time (minutes)'
]

df = df.drop(columns=cols_to_drop)
df['Order Placed At'] = pd.to_datetime(df['Order Placed At'])
df['Order Hour'] = df['Order Placed At'].dt.hour
df['Weekday'] = df['Order Placed At'].dt.dayofweek
df.drop(columns = ['Order Placed At'], inplace = True)

categorical_features = [
    'Restaurant name',
    'Restaurant ID',
    'Subzone',
    'Distance',
    'Order Hour',
    'Weekday'
]

df = pd.get_dummies(df, columns = categorical_features , drop_first = True)

# Clean column names to remove problematic characters for XGBoost
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)


df.dropna(inplace = True)

df['Items_in_order'] = df['Items_in_order'].apply(lambda x : x if x in top_15_items else 'Other')

df_main_items = df[df['Items_in_order'] != 'Other']
df_others = df[df['Items_in_order'] == 'Other']
n_samples = df_main_items.value_counts().max()
df_other_sampled = df_others.sample(n = n_samples, random_state = 42)

df = pd.concat([df_main_items, df_other_sampled])

X = df.drop('Items_in_order', axis = 1)
y_raw = df['Items_in_order']

le = LabelEncoder()
y = le.fit_transform(y_raw)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
logr = LogisticRegression()
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)
y_pred_label = le.inverse_transform(y_pred)
y_test_label = le.inverse_transform(y_test)

rf = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_rf_label = le.inverse_transform(y_pred_rf)

xg = XGBClassifier(n_estimators = 100, random_state = 42, use_label_encoder = False, eval_metric = 'mlogloss')
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)
y_pred_xg_label = le.inverse_transform(y_pred_xg)

class_rpt_rf = classification_report(y_test_label, y_pred_label)


print(f'Logisric Regression Report {class_rpt_rf}')


class_rpt_rf = classification_report(y_test_label, y_pred_rf_label)

print(f'Random Forest Report {class_rpt_rf}')



class_rpt_xg = classification_report(y_test_label, y_pred_xg_label)

print(f'XGBoost Report {class_rpt_xg}')

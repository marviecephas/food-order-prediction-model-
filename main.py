from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/order.csv'

df = pd.read_csv(file_path)

numerical_df = df.select_dtypes(include = 'number')
categorical_df = df.select_dtypes(include = 'object')
top_15_items = df['Items in order'].value_counts().nlargest(15).index
df_top_15 = df[df['Items in order'].isin(top_15_items)]

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


df.dropna(inplace = True)

print(df.head(50))

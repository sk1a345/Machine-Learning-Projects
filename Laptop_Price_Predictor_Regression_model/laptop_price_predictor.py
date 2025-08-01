from random import Random

import pandas as pd
import numpy as np

df = pd.read_csv("Laptop_Price_Predictor_Regression_model//laptop_data.csv")
# print(df.head())

# print(df.shape)

# Searching for the duplicates and dropping them
df.drop_duplicates(inplace=True)
# print(df.shape)

# print(df.info())

# We have the follwing columns:
# print(df.select_dtypes(include=['object','float','int64']).columns)

# ['Unnamed: 0', 'Company', 'TypeName', 'Inches', 'ScreenResolution',
#        'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']

# Dropping the Unnamed column:
df.drop(columns=['Unnamed: 0'],inplace=True)
# print(df.info())

# Removing the unnecessay stings from the RAM and WEIGHT
df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

# Changing the datatype
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

# print(df.info())

import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(df['Price'])

# Company
# df['Company'].value_counts().plot(kind='bar')

sns.barplot(x=df['Company'],y=df['Price'])
# plt.show()
#

# df['TypeName'].value_counts().plot(kind='bar')
# plt.show()

sns.barplot(x=df['TypeName'],y=df['Price'])
# plt.show()

sns.distplot(df['Inches'])
# plt.show()

sns.scatterplot(x=df['Inches'],y=df['Price'])
# plt.show()

# print(df['ScreenResolution'].value_counts())

df['TouchScreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
# print(df)
# df['TouchScreen'].value_counts().plot(kind='bar')
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# print(df['Ips'].value_counts())
# df['Ips'].value_counts().plot(kind='bar')
sns.barplot(x=df['Ips'],y=df['Price'])
# plt.show()

new =df['ScreenResolution'].str.split('x',n=1,expand=True)
print(new)
df['X_res'] = new[0]
df['Y_res'] = new[1]
# print(df.head())
# Remove commas, extract numbers, and keep only the first number
df['X_res'] = df['X_res'].str.replace(",", "", regex=False) \
                         .str.findall(r'(\d+\.?\d*)') \
                         .apply(lambda x: x[0] if x else None)
print(df['X_res'])
df['X_res'] = df['X_res'].astype('int32')
df['Y_res'] = df['Y_res'].astype("int32")
# print(df.head())
# print(df.info())

df['ppi'] = (((df['X_res']**2)+(df['Y_res']**2))/df['Price'])
print(df['ppi'])


print(df.info())
# print(df.corr()['Price'])
df.drop(columns=['ScreenResolution','Inches','X_res','Y_res'],inplace=True)

# print(df['Cpu'].value_counts())
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
# print(df['Cpu Name'].value_counts())

def fetch_processor(text):
    if text=='Intel Core i7' or text =='Intel Core i5' or text=='Intel Core i3':
        return text
    else:
        if text.split()[0] =='Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
print(df.head())

# df['Cpu brand'].value_counts().plot(kind='bar')

# sns.barplot(x=df['Cpu brand'],y=df['Price']).plot()

df.drop(columns=['Cpu','Cpu Name'],inplace=True)
# print(df.info())

# df['Ram'].value_counts().plot(kind='bar')
# sns.barplot(x=df['Ram'],y=df['Price']).plot()


df['Memory'] = df['Memory'].astype(str).replace(r"\.0",'',regex=True)
df['Memory'] = df['Memory'].str.replace("GB",'')
df['Memory'] = df['Memory'].str.replace('TB','000')
new = df['Memory'].str.split('+', n=1, expand=True)
df['first'] = new[0].str.strip()
df['second'] = new[1]

df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SDD'] = df['first'].apply(lambda x: 1 if "SSD" in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D','', regex=True)
df['second'] = df['second'].fillna('0')

df['Layer2HDD'] = df['second'].apply(lambda x: 1 if "HDD" in x else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if "SSD" in x else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D','', regex=True)

df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

df['HDD'] = df['first']*df['Layer1HDD'] + df['second']*df['Layer2HDD']
df['SSD'] = df['first']*df['Layer1SDD'] + df['second']*df['Layer2SSD']
df['Hybrid'] = df['first']*df['Layer1Hybrid'] + df['second']*df['Layer2Hybrid']
df['Flash_Storage'] = df['first']*df['Layer1Flash_Storage'] + df['second']*df['Layer2Flash_Storage']

df.drop(columns=[
    'first','second',
    'Layer1HDD','Layer1SDD','Layer1Hybrid','Layer1Flash_Storage',
    'Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'
], inplace=True)

# print(df.head())
# print(df.sample(5))

df.drop(columns=['Memory'],inplace=True)
# print(df.head())
# print(df.corr()['Price'])

df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
# print(df.head())

# print(df['Gpu'].value_counts())

df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
# print(df['Gpu brand'].value_counts())

df = df[df['Gpu brand']!='ARM']
# print(df.shape)

# print(df['Gpu brand'].value_counts())

df.drop(columns=['Gpu'],inplace=True)
# print(df.info())
# print(df['OpSys'].value_counts())

def cat_os(inp):
    if inp == 'Windows 10' or inp=='Windows 7' or inp=='Windows 10 S':
        return "Windows"
    elif inp == 'macOS' or inp =='Mac OS X':
        return "Mac"
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'],inplace=True)
# print(df.info())
# sns.distplot(df['Price']).plot()

# sns.distplot(np.log(df['Price'])).plot()

# Splitting the data into input and output data

x = df.drop(columns=['Price'])
y = np.log(df['Price'])

# print(x)
# print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=15,random_state=2)

# applying the OneHotEncoding:

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor

# from xgboost import XGBRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])

],remainder='passthrough')


'''
step2 = LinearRegression()
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
print("R2 score: ",r2_score(y_test,y_pred))
print("MAE: ",mean_absolute_error(y_test,y_pred))

'''

# Random -Forest REgressor:
step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
# print("R2 score: ",r2_score(y_test,y_pred))
# print("MAE : ",mean_absolute_error(y_test,y_pred))

# Exporting the model:

import pickle
# pickle.dump(df,open('df.pkl','wb'))
# pickle.dump(pipe,open('pipe.pkl','wb'))
#
# # C:\Users\HP\OneDrive\Machine_Learning_projects\df.pkl

with open(r'C:\Users\HP\OneDrive\Machine_Learning_projects\Laptop_Price_Predictor_Regression_model\df.pkl', 'wb') as f:
    pickle.dump(df, f)

with open(r'C:\Users\HP\OneDrive\Machine_Learning_projects\Laptop_Price_Predictor_Regression_model\pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)



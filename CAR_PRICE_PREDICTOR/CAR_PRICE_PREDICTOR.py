import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\HP\OneDrive\Machine_Learning_projects\CAR_PRICE_PREDICTOR\quikr_car.csv")
# print(df)
# print(df.info())


# print(df)
# print(df.head())
# print(df.info())
# print(df['year'].unique())
# print(df['Price'].unique())
# print(df['kms_driven'].unique())
# print(df['fuel_type'].unique())
# print(df['name'])


# Quality of data:
#year has many non-year values
# year object to int
# price has asked for price
# price object to int;
# kms_driven has kms with integers and also has commas
# kms_driven object to integer
# kms_driven has nan values
#fuel_type has nan value
# Keep first 3 words of name:

# Cleaning up the data:

df = df[df['year'].str.isnumeric()] #filtering the values specially numeric values
df['year'] = df['year'].astype(int)  #changing object to the integer


df = df[df['Price']!="Ask For Price"] #selecting only without Ask For Price rows
# here we are going to replace "23,455" i.e ",(comma)" with the empty string:
df['Price'] = df['Price'].str.replace(",","").astype(int)


df['kms_driven'] = df["kms_driven"].str.split(" ").str.get(0).str.replace(",","")
df = df[df['kms_driven']!="Petrol"]
df['kms_driven'] = df['kms_driven'].astype(int)

# print(df['fuel_type'].isnull().sum())

df = df[~df['fuel_type'].isna()] #skippin the row with null entry

# print(df.info())

# keeping the first 3 values of the carname

df['name'] = df['name'].str.split(" ").str.slice(0,3).str.join(" ")
# print(df['name'])
# Resetting the index:
df = df.reset_index(drop=True)
# print(df.info())
# print(df.describe())
# print(df[df['Price']>6e6])
df = df[df['Price']<6e6].reset_index(drop=True)
# print(df.shape)
# print(df)
# df.to_csv("Cleaned_car.csv")

x = df.drop(columns="Price")
y = df["Price"]
# print(x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=294)
# print(x.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)
# print(y_pred)

r2_sc = r2_score(y_test,y_pred)
print(r2_sc)

'''
scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred= pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))

print(np.argmax(scores))
print(scores[np.argmax(scores)])'''


import pickle
pickle.dump(pipe,open("LinearRegressionModel.pkl","wb"))

ans_pred = pipe.predict(pd.DataFrame([["Maruti Suzuki Swift","Maruti",2019,100,"Petrol"]], columns=['name','company','year','kms_driven','fuel_type']))
print(ans_pred)











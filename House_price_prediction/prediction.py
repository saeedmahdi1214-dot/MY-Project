import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("House price prediction/House Price Prediction Dataset.csv")
print(df)

print(df.info())
print(df.shape)
print(df.describe())

print(df.isnull().sum())


categorical_col = df.select_dtypes(include=['bool','category','object']).columns.tolist()
print(categorical_col)

df['Condition'] = df['Condition'].map({
    'Poor':0,
    'Fair':1,
    'Good':2,
    'Excellent':3
})

df['Garage'] = df['Garage'].map({
    'No':0,
    'Yes':1
})

df= pd.get_dummies(df,columns=['Location'],drop_first=True,dtype=int)


df = df.drop(columns='Id')

corr = df.corr(numeric_only = True)
print(corr)

sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',center=0)
plt.title('Heatmeap of correlation')
plt.show()

print(df.dtypes)

X = df.drop(columns='Price')
y = df['Price']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)

print(X_train)
print(y_train)

from sklearn.linear_model import LinearRegression

regressor =LinearRegression()
regressor.fit(X_train,y_train)

y_preds = regressor.predict(X_test)
print(y_preds)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae = mean_absolute_error(y_test,y_preds)
mse = mean_squared_error(y_test,y_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_preds)

print(f'Mean absolute error:{mae:.2f}')
print(f'Mean squared error:{mse:.2f}')
print(f'Root mean square error:{rmse}')
print(f'r2_score:{r2:.2f}')

import pickle

with open ('House_Price_Prediction.pkl','wb') as file:
    pickle.dump(regressor,file)

with open ('House_Price_Prediction.pkl','rb') as file:
    loaded_model = pickle.load(file)

prediction = loaded_model.predict(X_test.iloc[[0]])
print(prediction)

print(df.columns)
print(df.head())
print(df['Llocation'].unique())
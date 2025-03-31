import pandas as pd
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
def get_weekday(x):
    date1=x.split()[0]
    date2=datetime.strptime(date1,'%Y-%m-%d')
    week_day=date2.weekday()
    return week_day
'''

data=pd.read_csv('train.csv')
data['date']=data['datetime'].str.split(' ').str[0]
data['hour']=data['datetime'].str.split(' ').str[1].str.split(':').str[0].astype(int)
data['year']=data['date'].str.split('-').str[0].astype(int)
data['month']=data['date'].str.split('-').str[1].astype(int)
data['day']=data['date'].str.split('-').str[2].astype(int)
data['weekday']=data['date'].apply(get_weekday)
data['temp']=data['temp'].astype(int)
data['atemp']=data['atemp'].astype(int)
data['humidity']=data['humidity'].astype(int)
data['windspeed']=data['windspeed'].astype(int)

fig,axes=plt.subplots(2,2,figsize=(15,20))
axes=axes.flatten()
columns_to_plot = ['temp', 'atemp', 'humidity', 'windspeed']
for i, column in enumerate(columns_to_plot):
    value_counts = data[column].value_counts().sort_index()
    value_counts.plot(kind='bar', ax=axes[i], title=column.capitalize())

# 添加标题和标签
for ax in axes:
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('Count')
    ax.set_title(ax.get_title())  # 确保标题格式一致



windspeed0=data[data['windspeed']==0].copy()
windspeed1=data[data['windspeed']!=0]
model_wind=RandomForestRegressor(n_estimators=1000, random_state=42)
X=windspeed1[['temp','atemp','humidity','year','month','day','hour','weekday']]
y=windspeed1['windspeed']
model_wind.fit(X,y)
windspeed0['windspeed']=model_wind.predict(windspeed0[['temp','atemp','humidity','year','month','day','hour','weekday']])

data = pd.concat([windspeed0, windspeed1], ignore_index=True)
data['windspeed']=data['windspeed'].astype(int)
data.to_csv('data.csv',index=False)
'''
data=pd.read_csv('data.csv')
'''
sns.pairplot(data,x_vars=['holiday','workingday','weather','season','weekday','hour','windspeed','humidity','temp','atemp'],y_vars=['casual','registered','count'],plot_kws={'alpha': 0.1})
plt.savefig('pairplot.png')
plt.show()
'''
data.drop(['datetime','date'],axis=1,inplace=True)
'''
season=data.groupby('holiday').agg(
    {'casual':'sum','registered':'sum','count':'sum'}
)

season.plot(kind='bar',figsize=(10,5),title='season')
'''
character=['temp','atemp','humidity','windspeed','hour','weekday']
target=['registered']
X1=data[character]
y=data[target]
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

model=RandomForestRegressor(n_estimators=1000, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练线性回归模型
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R² Score: {r2}')
X=pd.read_csv('test.csv')

X['date']=X['datetime'].str.split(' ').str[0]
X['hour']=X['datetime'].str.split(' ').str[1].str.split(':').str[0].astype(int)
X['year']=X['date'].str.split('-').str[0].astype(int)
X['month']=X['date'].str.split('-').str[1].astype(int)
X['day']=X['date'].str.split('-').str[2].astype(int)
X['weekday']=X['date'].apply(get_weekday)
X['temp']=X['temp'].astype(int)
X['atemp']=X['atemp'].astype(int)
X['humidity']=X['humidity'].astype(int)
X['windspeed']=X['windspeed'].astype(int)
X1=X[character]
x_scaled=scaler.transform(X1)
y_pred=model.predict(x_scaled)
X['registered']=y_pred

X.to_csv('test.csv',index=False)    





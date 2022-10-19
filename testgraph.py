import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import cross_val_score
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
sns.set_style("whitegrid")

flights=pd.read_excel('./Data_Train.xlsx')
# print(flights.head())

# flights.info()

flights.dropna(inplace=True)
# flights.info()

flights['Date_of_Journey']=pd.to_datetime(flights['Date_of_Journey'])
flights['Dep_Time']=pd.to_datetime(flights['Dep_Time'],format='%H:%M').dt.time
flights['Additional_Info']=flights['Additional_Info'].str.replace('No info','No Info')
flights['Duration']=flights['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
flights['Duration']=pd.to_numeric(flights['Duration'])
flights['weekday']=flights[['Date_of_Journey']].apply(lambda x:x.dt.day_name()) #bug
flights["month"] = flights['Date_of_Journey'].map(lambda x: x.month_name()) #bug
flights['Dep_Time']=flights['Dep_Time'].apply(lambda x:x.hour)
flights['Dep_Time']=pd.to_numeric(flights['Dep_Time'])

# print(flights['Date_of_Journey'])

flights.drop(['Route','Arrival_Time','Date_of_Journey'],axis=1,inplace=True)
# print(flights.head())

var_mod = ['Airline','Source','Destination','Additional_Info','Total_Stops','weekday','month','Dep_Time']
le = LabelEncoder()
for i in var_mod:
    flights[i] = le.fit_transform(flights[i])
# print(flights.corr())
# sns.heatmap(flights.corr(),cmap='coolwarm',annot=True)
# plt.show()

# outlier
def outlier(df):
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR= Q3-Q1
        LE=Q1-1.5*IQR
        UE=Q3+1.5*IQR
        df[i]=df[i].mask(df[i]<LE,LE)
        df[i]=df[i].mask(df[i]>UE,UE)
    return df
flights=outlier(flights)
x=flights.drop('Price',axis=1)
y=flights['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

rfr=RandomForestRegressor(n_estimators=500) #random data 500 item
rfr.fit(x_train,y_train)
features=x.columns
importances = rfr.feature_importances_
indices = np.argsort(importances)
# plt.figure(1)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), features[indices])
# plt.xlabel('Relative Importance')
# plt.show()

predictions=rfr.predict(x_test)
# print(predictions)
# predictions=rfr.predict(x_test)
# plt.scatter(y_test,predictions)
# plt.show()

# print('MAE:', metrics.mean_absolute_error(y_test, predictions))
# print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# print('r2_score:', (metrics.r2_score(y_test, predictions)))

# regg=[LinearRegression(),RandomForestRegressor(),SVR(),DecisionTreeRegressor()]
# mean=[]
# std=[]
# for i in regg:
#     cvs=cross_val_score(i,x,y,cv=5,scoring=make_scorer(r2_score))
#     mean.append(np.mean(cvs))
#     std.append(np.std(cvs))
# for i in range(4):
#     print(regg[i].__class__.__name__,':',mean[i])

test_file=pd.read_excel('Test_set.xlsx')
test_file['Date_of_Journey']=pd.to_datetime(test_file['Date_of_Journey'])
test_file['Dep_Time']=pd.to_datetime(test_file['Dep_Time'],format='%H:%M').dt.time
test_file['Duration']=test_file['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
test_file['Duration']=pd.to_numeric(test_file['Duration'])
test_file['Dep_Time']=test_file['Dep_Time'].apply(lambda x:x.hour)
test_file['Dep_Time']=pd.to_numeric(test_file['Dep_Time'])
test_file["month"] = test_file['Date_of_Journey'].map(lambda x: x.month_name())
test_file['weekday']=test_file[['Date_of_Journey']].apply(lambda x:x.dt.day_name())
test_file['Additional_Info']=test_file['Additional_Info'].str.replace('No info','No Info')
test_file.drop(['Date_of_Journey','Route','Arrival_Time'],axis=1,inplace=True)
for i in var_mod:
    test_file[i]=le.fit_transform(test_file[i])
test_price_predictions=rfr.predict(test_file)
print(test_price_predictions)

# info  = []
# info[0] = input("Date (yyyy-mm-dd) = ")
# info[1] = input(" Time (hr:min:sec) = ")
# info[2] = input("Duration time (ex. 10h 55m)").str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
# info[2] = pd.to_numeric(info[2])
# info[1] = info[1].apply(lambda x:x.hour)

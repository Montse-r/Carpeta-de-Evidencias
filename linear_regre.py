import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
plt.scatter(X,Y,color = 'red')
plt.plot(X, regressor.predict(X),color ='blue')
plt.show()

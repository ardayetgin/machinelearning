from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
boston = datasets.load_boston()

boston.keys()

print(boston.feature_names)

boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['MEDV'] = boston.target

print(boston_df.head())

plt.scatter(boston_df['LSTAT'],boston_df['MEDV'], alpha=0.7)
plt.xlabel('LSTAT')
plt.ylabel('medv')
plt.show()

lstat = boston_df[['LSTAT']].values
medv = boston_df[['MEDV']].values

model = linear_model.LinearRegression()
model.fit(lstat,medv)

plt.scatter(lstat,medv, color='blue', alpha=0.7)
plt.plot(lstat,model.predict(lstat), color='red')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

model.intercept_

model.coef_

x = boston_df[['LSTAT', 'AGE']].values
y = boston_df['MEDV'].values
model2 = linear_model.LinearRegression()
model2.fit(x,y)

model2.intercept_

model2.coef_

x = boston_df.drop('MEDV', axis=1)
y = boston_df['MEDV']

model = linear_model.LinearRegression()
model.fit(x,y)


model.intercept_

model.coef_

import seaborn as sns

car_crashes = sns.load_dataset('car_crashes')
sns.lmplot(x='alcohol', y='total', data=car_crashes)
plt.show()

iris = sns.load_dataset('iris')
sns.lmplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.show()

iris = sns.load_dataset('iris')
sns.lmplot(x='sepal_length', y='sepal_width', col='species', data=iris)
plt.show()

iris = sns.load_dataset('iris')
sns.lmplot(x='sepal_length', y='sepal_width', row='species', data=iris)
plt.show()

iris = sns.load_dataset('iris')
sns.residplot(x='sepal_length', y='sepal_width', data=iris, color='red')
plt.show()


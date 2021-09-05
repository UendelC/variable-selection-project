import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_data = pd.read_csv('projeto-2/dados/X_train.csv', sep=',', header=None)
y_data = pd.read_csv('projeto-2/dados/y_train.csv', sep=',', header=None)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

# n_jobs é um parâmetro que diz respeito a quantidade de threads a serem utilizadas, se -1, usa todas as disponíveis
trees = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1)

trees.fit(X_train, y_train)

y_predict = trees.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_predict))
#%%

from tensorflow import keras
from keras.models import Sequential
from keras.layers import InputLayer, Dense
import pandas as pd
import numpy as np

#%%

#Importing data
train_db = pd.read_csv('../pre-processamento-e-transformacao/TRAIN_DB.csv')
test_db = pd.read_csv('../pre-processamento-e-transformacao/TEST_DB.csv')

y_train = train_db['SalePrice']
x_train = train_db.drop(['SalePrice'], axis=1)

y_test = test_db['SalePrice']
x_test = test_db.drop(['SalePrice'], axis=1)

# Transforming target variable interval
y_train = np.expm1((y_train + abs(y_train.min())) / 181000)
y_test = np.expm1((y_test + abs(y_test.min())) / 180000)

# Normalizar
from sklearn.preprocessing import MinMaxScaler

#
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from keras.callbacks import EarlyStopping

epochs = 500
batch_size = 40
callbacks = [EarlyStopping(monitor='loss', patience=20, verbose=1)]

# neurons = [1, 2, 3, 4, 5, 6, 7]
hidden_neurons_1 = [10, 100, 200, 300]
hidden_neurons_2 = [10, 100, 200, 300]
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.009]
result = []
result_log = []
choose_model = []

ds_len = np.size(x_train, 1)


from sklearn.metrics import median_absolute_error, r2_score

y_train_log = np.log1p(y_train)

for n in hidden_neurons_1:
    for m in hidden_neurons_2:
        mlp = Sequential([
            InputLayer(ds_len),
            Dense(n, activation="relu"),
            Dense(m, activation="relu"),
            Dense(1)
        ])

        for lr in lrs:
            opt = keras.optimizers.Adam(learning_rate=lr)
            mlp.compile(optimizer=opt, loss="mse")

            Seq_without_y_log = mlp.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                        verbose=0)
            y_pred = mlp.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mae = median_absolute_error(y_test, y_pred)
            result.append([n, m, lr, r2, mae])
            print(['Sem Log', n, m, lr, r2, mae])

            Seq_with_y_log = mlp.fit(x_train, y_train_log, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                     verbose=0)
            y_pred_log = np.expm1(mlp.predict(x_test))
            r2_log = r2_score(y_test, y_pred_log)
            mae_log = median_absolute_error(y_test, y_pred_log)
            result_log.append([n, m, lr, r2_log, mae_log])
            print(['Com Log', n, m, lr, r2_log, mae_log])

print(result)
print("\n\n")
print(result_log)




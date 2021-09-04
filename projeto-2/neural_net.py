import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import sklearn.metrics as metrics

from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, InputLayer


x_data = pd.read_csv('projeto-2/dados/X_train.csv', sep=',', header=None)
y_data = pd.read_csv('projeto-2/dados/y_train.csv', sep=',', header=None)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crie um modelo de redes neurais usando o Keras;

input_neuron, hidden_neuron, output_neuron = 8, 10, 1

mlp = Sequential()
mlp.add(InputLayer(input_shape=(input_neuron,)))
mlp.add(Dense(hidden_neuron, activation='sigmoid'))
mlp.add(Dense(output_neuron, activation='sigmoid'))

# compilo o modelo

mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# treine o modelo

batch_size = 10
Log = mlp.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test))

# Visualize o comportamento do erro durante o treinamento de forma gráfica
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(Log.history['loss'], label='Erro no treinamento')
ax.legend()

# Calcule a acurácia do conjunto de teste

y_pred = mlp.predict(X_test)

y_pred_bin = np.around(y_pred)

accuracy_score(y_test, y_pred_bin)

# gerar matriz de confusão

cm = confusion_matrix(y_test, y_pred_bin)
df_cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])

# Recuperar os valores da matriz de confusão
# Verdadeiro Positivo
a = df_cm.iloc[0, 0]
# Falso Negativo
b = df_cm.iloc[1, 0]
# Verdadeiro Negativo
c = df_cm.iloc[0, 1]
# Falso Positivo
d = df_cm.iloc[1, 1]

# calculando a acurácia do modelo
acuracia = (a + d) / (a + b + c + d)

# calculando a precisão
precision = a / (a + b)

# calculando a especificidade do modelo
specificity = d / (b + d)

# calculando a sensibilidade do modelo
sensitivity = a / (a + c)

# calculando a área sob a curva ROC
area_roc = roc_auc_score(y_test, y_pred)

# gerando o gráfico da área sob a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='Área sob a curva ROC = %0.2f' % area_roc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Área sob a curva ROC')
plt.legend(loc="lower right")
plt.show()

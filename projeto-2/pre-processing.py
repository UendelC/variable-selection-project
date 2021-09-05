import numpy as np
import pandas as pd

transact_train_database = pd.read_csv('projeto-2/data/transact_train.txt', sep='|')
transact_train_database.sample(5)

transact_test_database = pd.read_csv('projeto-2/data/transact_class.txt', sep = '|')
transact_test_database.sample(3)

# ------------
# Alterando a Granularidade

# alterando a granularidade da base de dados
# remove todas as linhas com valores de sessionNo iguais exceto a última
transact_train_database = transact_train_database.drop_duplicates(subset=['sessionNo'], keep='last')

# separa variável alvo no conjunto de treinamento
transact_train_database_y = transact_train_database['order']
# replace y por 1 e n por 0
transact_train_database_y = transact_train_database_y.replace(to_replace=['y', 'n'], value=[1, 0])

# remove a coluna alvo do conjunto de treinamento
transact_train_database_x = transact_train_database.drop(['order'], axis=1)


# ----------
# Trata missing values


import csv
import random
import numpy as np
import pandas as pd

# Mudando a granularidade

TRN_Original = pd.read_csv("data/transact_train.txt", sep="|")
TST_Original = pd.read_csv("data/transact_class.txt", sep="|")

sessionIDs_TRN = TRN_Original['sessionNo']
sessionIDs_TST = TST_Original['sessionNo']


# Normalização das variáveis

# São criadas duas funções para auxiliar esse processo

# Função que calcula o mínimo e máximo de cada variável
def get_Min_Max(X):
    '''
    Input: 
        X - Data frame
    Output:
        result - Um dicionário onde a chave é o nome da coluna de X e o valor é uma tupla (Min, Max)
    '''
    result = {}
    for v in X.columns:
        result[v] = (np.min(X[v]), np.max(X[v]))
    return result


# Função que normaliza os dados entre zero e um usando a fórmula (valor_atual - min) / (max - min)
def normalize(X, MinMax):
    '''
    Input:
        X - Data frame
        MinMax -  Um dicionário contendo o mínimo e máximo de uma variável. Ele ser um parâmetro permite que seja
                   aplicado em conjuntos diferentes
    Output:
        result - Um Data frame normalizado
    '''

    result = X.copy()

    for v in MinMax:

        min_v, max_v = MinMax[v]

        div = max_v - min_v

        if div == 0:
            div = 1

        result[v] = (X[v] - min_v) / div

        # Correção para aplicação em conjuntos de testes com valores diferentes do conjunto de treinamento

        # valores menores que zero são setados para zero
        idx_0 = result[v] < 0
        result.loc[idx_0, v] = 0

        # valores maiores que um são setados para um
        idx_1 = result[v] > 1
        result.loc[idx_1, v] = 1

    return result


# Apoio ao tratamento de missing values

l_varNumeric = ['cMinPrice', 'cMaxPrice', 'cSumPrice', 'bMinPrice', 'bMaxPrice', 'bSumPrice', 'bStep', 'maxVal',
                'customerScore', 'accountLifetime', 'payments', 'age', 'address', 'lastOrder']
l_varString = ['availability', 'onlineStatus']


def replaceValueMissing(vl, listVar, data):
    for v in listVar:
        rows = data[v] == vl
        data.loc[rows, v] = np.NaN


def convertFloat(listVar, data):
    for v in listVar:
        data[v] = data[v].astype(float)


def replaceMissingByMean(listVar, data):
    for v in listVar:
        avg = data[v].mean(axis=0)
        data[v].fillna(avg, inplace=True)


def replaceMissingByFixedValue(vl, listVar, data):
    for v in listVar:
        data[v].fillna(vl, inplace=True)


# ----------------------------------------------------------------

# ----------- Mudando a granularidade ----------- 

def getSessionWithTransaction(ReplicatedSession):
    idx_TransactionSession = {s: [] for s in set(ReplicatedSession)}
    # {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for i in range(len(ReplicatedSession)):
        s = ReplicatedSession[i]
        idx_TransactionSession[s].append(i)

    # {0: [0, 1, 2, 3, 4], 1: [5, 6, 7], 2: [8, 9, 10], 3: [11, 12, 13, 14, 15, 16, 17], 4: [18, 19],
    # 5: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 6: [30, 31, 32, 33, 34, 35, 36, 37],
    # 7: [38, 39, 40, 41, 42], 8: [43, 44, 45, 46, 47, 48, 49], 9: [50, 51, 52, 53]}

    return idx_TransactionSession


idx_SessionWithTransaction_TRN = getSessionWithTransaction(sessionIDs_TRN)
idx_SessionWithTransaction_TST = getSessionWithTransaction(sessionIDs_TST)

idx_lastTransactionBySession_TRN = [np.max(x) for x in idx_SessionWithTransaction_TRN.values()]
idx_lastTransactionBySession_TST = [np.max(x) for x in idx_SessionWithTransaction_TST.values()]

TRN_X = TRN_Original.iloc[idx_lastTransactionBySession_TRN, :-1]
TRN_Y = TRN_Original.iloc[idx_lastTransactionBySession_TRN, -1]

TST_X_tmp = TST_Original.iloc[idx_lastTransactionBySession_TST, :]
TST_X = TST_X_tmp.copy()
# TST_Y = TST_Y_Original [' prediction ']

# print(TST_X)

TRN_X.to_csv('data/granularityOk.csv')

# ----------- Missing values ----------- 

rows_TRN = TRN_X['customerNo'] == "?"
TRN_X.loc[rows_TRN, 'maxVal'] = 0
TRN_X.loc[rows_TRN, 'customerScore'] = 0
TRN_X.loc[rows_TRN, 'accountLifetime'] = 0
TRN_X.loc[rows_TRN, 'payments'] = 0
TRN_X.loc[rows_TRN, 'age'] = 0
TRN_X.loc[rows_TRN, 'address'] = 0
TRN_X.loc[rows_TRN, 'lastOrder'] = 0

rows_TST = TST_X['customerNo'] == "?"
TST_X.loc[rows_TST, 'maxVal'] = 0
TST_X.loc[rows_TST, 'customerScore'] = 0
TST_X.loc[rows_TST, 'accountLifetime'] = 0
TST_X.loc[rows_TST, 'payments'] = 0
TST_X.loc[rows_TST, 'age'] = 0
TST_X.loc[rows_TST, 'address'] = 0
TST_X.loc[rows_TST, 'lastOrder'] = 0

replaceValueMissing('?', l_varNumeric, TRN_X)
replaceValueMissing('?', l_varString, TRN_X)
replaceValueMissing('?', l_varNumeric, TST_X)
replaceValueMissing('?', l_varString, TST_X)

TRN_X.to_csv('data/beforeReplace.csv')

convertFloat(l_varNumeric, TRN_X)
convertFloat(l_varNumeric, TST_X)

replaceMissingByMean(l_varNumeric, TRN_X)
replaceMissingByMean(l_varNumeric, TST_X)

replaceMissingByFixedValue('ausente ', l_varString, TRN_X)
replaceMissingByFixedValue('ausente ', l_varString, TST_X)

TRN_X.to_csv('data/finalFile.csv')

# bSumPrice_cSumPrice       - (float) Razão entre média da soma dos valores na cesta e média da soma dos valores clicados. Tem-se assim a porcentagem do valor total clicado que realmente interessa ao cliente.
# wasLogged                 - (boolean) Mostra se o usuário logou durante a sessão
# lastOrder_accountLifetime - (float) Razão entre a média da soma dos dias desde a última compra e o tempo de vida da conta. Tem-se assim a porcentagem de dias de compra.
# onlineTime                - (float) Tempo que o cliente permaneceu online durante a sessão
# bCount_cCount             - (float) Razão entre a soma da quantidade de produtos na cesta e a quantidade de produtos clicados

replaceValueMissing('?', l_varNumeric, TRN_Original)
replaceValueMissing('?', l_varString, TRN_Original)
replaceValueMissing('?', l_varNumeric, TST_Original)
replaceValueMissing('?', l_varString, TST_Original)
convertFloat(l_varNumeric, TRN_Original)
convertFloat(l_varNumeric, TST_Original)

# --------- Definindo bSumPrice_cSumPrice ---------
TRN_X = TRN_X.set_index('sessionNo')
TST_X = TST_X.set_index('sessionNo')

TST_X['bSumPrice_cSumPrice'] = TST_Original.groupby('sessionNo')['bSumPrice'].mean() / \
                               TST_Original.groupby('sessionNo')['cSumPrice'].mean()
TRN_X['bSumPrice_cSumPrice'] = TRN_Original.groupby('sessionNo')['bSumPrice'].mean() / \
                               TRN_Original.groupby('sessionNo')['cSumPrice'].mean()

# --------- Definindo wasLogged ---------
# TST_X['wasLogged'] = TRN_Original.groupby('sessionNo')['customerNo'].apply(lambda x: (set(x)) != '?')
# TRN_X['wasLogged'] = TRN_Original.groupby('sessionNo')['customerNo'].apply(lambda x: (set(x)) != '?')

# --------- Definindo lastOrder_accountLifetime ---------
TST_X['lastOrder_accountLifetime'] = TST_Original.groupby('sessionNo')['lastOrder'].max() / \
                                     TST_Original.groupby('sessionNo')['accountLifetime'].max()
TRN_X['lastOrder_accountLifetime'] = TRN_Original.groupby('sessionNo')['lastOrder'].max() / \
                                     TRN_Original.groupby('sessionNo')['accountLifetime'].max()

# --------- Definindo onlineTime ---------
# TRN_Original_copy = TRN_Original
# TRN_Original_copy.loc[TRN_Original_copy['onlineStatus'] != 'y', 'duration'] = 0
# print(TRN_Original_copy.groupby('sessionNo')['duration'].sum())
# print(TRN_Original.groupby('sessionNo')['onlineStatus'].agg({lambda x: TRN_Original.loc[x.index, 'duration'][x == 'y']}))

# print(TRN_Original.groupby('sessionNo').apply({lambda x: x[x['onlineStatus'] == 'y']['duration'].sum()}))

# TRN_Original.groupby('sessionNo').aggregate({lambda x: print(x.values)})

# print(TRN_Original_copy.groupby('sessionNo')['duration'].diff())

TRN_Grouped_by_sessionNo = TRN_Original.groupby('sessionNo')
duration_sum = 0
index = 1
rows_it = 0

TRN_X['onlineTime'] = random.randint(0, len(TRN_X))

for index, row in TRN_Grouped_by_sessionNo:
    rows_len = len(row['duration'])
    duration_sum = 0
    changed_id = True

    for it in range(rows_it, rows_len + rows_it):
        if row['onlineStatus'][it] == 'y':
            duration_sum += (row['duration'][it] - row['duration'][it - 1]) if not changed_id else row['duration'][it]

        if changed_id:
            changed_id = False

        rows_it += 1

    TRN_X.loc[index, 'onlineTime'] = duration_sum
    index += 1

# --------- Definindo bCount_cCount ---------
TST_X['bCount_cCount'] = TST_Original.groupby('sessionNo')['bCount'].sum() / TST_Original.groupby('sessionNo')[
    'cCount'].sum()
TRN_X['bCount_cCount'] = TRN_Original.groupby('sessionNo')['bCount'].sum() / TRN_Original.groupby('sessionNo')[
    'cCount'].sum()

# --------- Transformação ---------
TRN_X = pd.get_dummies(TRN_X, prefix_sep='_')
TST_X = pd.get_dummies(TST_X, prefix_sep='_')

norm_min_max = get_Min_Max(TRN_X)
TRN_X = normalize(TRN_X, norm_min_max)
TST_X = normalize(TST_X, norm_min_max)

threshold_var = 0

l_var = [x for x in TRN_X.columns if TRN_X[x].var() <= threshold_var]

for v in l_var:
    TRN_X = TRN_X.drop([v], axis=1)
    TST_X = TST_X.drop([v], axis=1)

TRN_X.to_csv('TRN_X.csv ', index=False)
TRN_Y.to_csv('TRN_Y.csv ', index=False)
TST_X.to_csv('TST_X.csv ', index=False)
TST_Y.to_csv('TST_Y.csv ', index=False)

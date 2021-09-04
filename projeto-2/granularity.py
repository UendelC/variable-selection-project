import csv
import random
import numpy as np
import pandas as pd

# Mudando a granularidade

TRN_Original = pd.read_csv("data/transact_train.csv")
TST_Original = pd.read_csv("data/transact_class.csv")

sessionIDs_TRN = TRN_Original ['sessionNo']
sessionIDs_TST = TST_Original ['sessionNo']

# Apoio ao tratamento de missing values

l_varNumeric = ['cMinPrice', 'cMaxPrice', 'cSumPrice', 'bMinPrice', 'bMaxPrice', 'bSumPrice', 'bStep', 'maxVal', 'customerScore', 'accountLifetime', 'payments', 'age', 'address', 'lastOrder']
l_varString = ['availability', 'onlineStatus']

def replaceValueMissing (vl , listVar , data ):
    for v in listVar:
        rows = data[v] == vl
        data.loc[rows , v] = np.NaN

def convertFloat ( listVar , data ):
    for v in listVar :
        data [v] = data [v]. astype ( float )

def replaceMissingByMean ( listVar , data ):
    for v in listVar :
        avg = data [v]. mean ( axis =0)
        data [v]. fillna (avg , inplace = True )

def replaceMissingByFixedValue (vl , listVar , data ):
    for v in listVar :
        data [v]. fillna (vl , inplace = True )
# ----------------------------------------------------------------

# ----------- Mudando a granularidade ----------- 

def getSessionWithTransaction ( ReplicatedSession ):
    idx_TransactionSession = {s:[] for s in set ( ReplicatedSession )}

    for i in range ( len ( ReplicatedSession )):
        s = ReplicatedSession [i]
        idx_TransactionSession [s]. append (i)

    return idx_TransactionSession

idx_SessionWithTransaction_TRN = getSessionWithTransaction ( sessionIDs_TRN )
idx_SessionWithTransaction_TST = getSessionWithTransaction ( sessionIDs_TST )

idx_lastTransactionBySession_TRN =[np.max(x) for x in idx_SessionWithTransaction_TRN . values()]
idx_lastTransactionBySession_TST =[np.max(x) for x in idx_SessionWithTransaction_TST . values()]

TRN_X = TRN_Original . iloc [ idx_lastTransactionBySession_TRN ,:-1]
TRN_Y = TRN_Original . iloc [ idx_lastTransactionBySession_TRN ,-1]

TST_X_tmp = TST_Original . iloc [ idx_lastTransactionBySession_TST ,:]
TST_X = TST_X_tmp . copy()
# TST_Y = TST_Y_Original [' prediction ']

# print(TST_X)

TRN_X.to_csv( 'data/granularityOk.csv' )

# ----------- Missing values ----------- 

rows_TRN = TRN_X ['customerId']=="?"
TRN_X .loc [ rows_TRN , 'maxVal'] = 0
TRN_X .loc [ rows_TRN , 'customerScore'] = 0
TRN_X .loc [ rows_TRN , 'accountLifetime'] = 0
TRN_X .loc [ rows_TRN , 'payments'] = 0
TRN_X .loc [ rows_TRN , 'age'] = 0
TRN_X .loc [ rows_TRN , 'address'] = 0
TRN_X .loc [ rows_TRN , 'lastOrder'] = 0

rows_TST = TST_X ['customerId']=="?"
TST_X .loc [ rows_TST , 'maxVal'] = 0
TST_X .loc [ rows_TST , 'customerScore'] = 0
TST_X .loc [ rows_TST , 'accountLifetime'] = 0
TST_X .loc [ rows_TST , 'payments'] = 0
TST_X .loc [ rows_TST , 'age'] = 0
TST_X .loc [ rows_TST , 'address'] = 0
TST_X .loc [ rows_TST , 'lastOrder'] = 0

replaceValueMissing ('?', l_varNumeric , TRN_X )
replaceValueMissing ('?', l_varString , TRN_X )
replaceValueMissing ('?', l_varNumeric , TST_X )
replaceValueMissing ('?', l_varString , TST_X )

TRN_X.to_csv( 'data/beforeReplace.csv' )

convertFloat ( l_varNumeric , TRN_X )
convertFloat ( l_varNumeric , TST_X )

replaceMissingByMean ( l_varNumeric , TRN_X )
replaceMissingByMean ( l_varNumeric , TST_X )

replaceMissingByFixedValue ('ausente ', l_varString , TRN_X )
replaceMissingByFixedValue ('ausente ', l_varString , TST_X )

TRN_X.to_csv( 'data/finalFile.csv' )

# bSumPrice_cSumPrice       - (float) Razão entre média da soma dos valores na cesta e média da soma dos valores clicados. Tem-se assim a porcentagem do valor total clicado que realmente interessa ao cliente.
# wasLogged                 - (boolean) Mostra se o usuário logou durante a sessão
# lastOrder_accountLifetime - (float) Razão entre a média da soma dos dias desde a última compra e o tempo de vida da conta. Tem-se assim a porcentagem de dias de compra.
# onlineTime                - (float) Tempo que o cliente permaneceu online durante a sessão
# bCount_cCount             - (float) Razão entre a soma da quantidade de produtos na cesta e a quantidade de produtos clicados

replaceValueMissing ('?', l_varNumeric , TRN_Original )
replaceValueMissing ('?', l_varString , TRN_Original )
replaceValueMissing ('?', l_varNumeric , TST_Original )
replaceValueMissing ('?', l_varString , TST_Original )
convertFloat ( l_varNumeric , TRN_Original )
convertFloat ( l_varNumeric , TST_Original )

# --------- Definindo bSumPrice_cSumPrice ---------
TRN_X = TRN_X.set_index ('sessionNo')
TST_X = TST_X.set_index ('sessionNo')

TST_X ['bSumPrice_cSumPrice'] = TST_Original.groupby('sessionNo')['bSumPrice'].mean() / TST_Original.groupby('sessionNo')['cSumPrice'].mean()
TRN_X ['bSumPrice_cSumPrice'] = TRN_Original.groupby('sessionNo')['bSumPrice'].mean() / TRN_Original.groupby('sessionNo')['cSumPrice'].mean()


# --------- Definindo wasLogged ---------
TST_X ['wasLogged'] = TRN_Original.groupby('sessionNo')['customerId'].apply(lambda x: (set(x)) != '?')
TRN_X ['wasLogged'] = TRN_Original.groupby('sessionNo')['customerId'].apply(lambda x: (set(x)) != '?')

# --------- Definindo lastOrder_accountLifetime ---------
TST_X ['lastOrder_accountLifetime'] = TST_Original.groupby('sessionNo')['lastOrder'].max() / TST_Original.groupby('sessionNo')['accountLifetime'].max()
TRN_X ['lastOrder_accountLifetime'] = TRN_Original.groupby('sessionNo')['lastOrder'].max() / TRN_Original.groupby('sessionNo')['accountLifetime'].max()

print(TRN_X ['lastOrder_accountLifetime'])
import csv
import random
import numpy as np
import pandas as pd

# Mudando a granularidade

TRN_Original = pd.read_csv("data/transact_train.csv")

sessionIDs_TRN = TRN_Original ['sessionNo']

def getSessionWithTransaction ( ReplicatedSession ):
    idx_TransactionSession = {s:[] for s in set ( ReplicatedSession )}

    for i in range ( len ( ReplicatedSession )):
        s = ReplicatedSession [i]
        idx_TransactionSession [s]. append (i)

    return idx_TransactionSession

idx_SessionWithTransaction_TRN = getSessionWithTransaction ( sessionIDs_TRN )

idx_lastTransactionBySession_TRN =[np.max(x) for x in idx_SessionWithTransaction_TRN . values()]

TRN_X = TRN_Original . iloc [ idx_lastTransactionBySession_TRN ,:-1]

print(TRN_X)
import numpy as np
import pandas as pd

ic_house_pred_train = pd.read_csv('data/ic_house_pred_train.csv')
train_db = ic_house_pred_train.drop(['Id'], axis=1)

numeric_columns = train_db.select_dtypes(include=np.number).columns
categoric_columns = [x for x in train_db.columns if x not in numeric_columns]

print("Numeric missing values from train database")
print(train_db[numeric_columns].isnull().sum().sum())

print("Categoric missing values from train database")
print(train_db[categoric_columns].isnull().sum().sum())

def replaceMissingValuesByMean(var_list, data):
    for var in var_list:
        avg = data[var].mean(axis=0)
        data[var].fillna(avg, inplace=True)

# Replacing numeric columns
replaceMissingValuesByMean(numeric_columns, train_db)

counts  = train_db['Street'].value_counts(normalize=True)
mask = train_db.isin(counts[counts > 0.9].index)
print(mask)
print(train_db[mask])
print( pd.get_dummies(train_db[mask], prefix_sep='_') )

# for x in train_db.columns:
#    if  x not in numeric_columns:
#        counts  = train_db[x].value_counts(normalize=True)
        # print(counts)
#        mask = train_db.isin(counts[counts > 0.9].index)
#        print( mask )
        # print( pd.get_dummies(train_db[mask], prefix_sep='_') )
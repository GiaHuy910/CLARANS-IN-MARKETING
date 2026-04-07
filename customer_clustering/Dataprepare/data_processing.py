#đây mới là file để lấy data
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path=r'../customer_clustering/datasets/Customer_Data_2k.csv'
data=pd.read_csv(path)

#tiền xử lí dữ liệu
data.dropna(inplace=True)

columns_to_drop = ['cust_id']
data = data.drop(columns=columns_to_drop)

features_scaling = [
    'balance',
    'balance_frequency',
    'purchases',
    'oneoff_purchases',
    'installments_purchases',
    'cash_advance',
    'purchases_frequency',
    'oneoff_purchases_frequency',
    'purchases_installments_frequency',
    'cash_advance_frequency',
    'cash_advance_trx',
    'purchases_trx',
    'credit_limit',
    'payments',
    'minimum_payments',
    'prc_full_payment',
    'tenure'
]
features_IQR = [
    'balance',
    'purchases',
    'oneoff_purchases',
    'installments_purchases',
    'cash_advance',
    'oneoff_purchases_frequency',
    'purchases_installments_frequency',
    'cash_advance_frequency',
    'cash_advance_trx',
    'purchases_trx',
    'credit_limit',
    'payments',
    'minimum_payments',
    'tenure'
]
#Log
for col in features_scaling:
    data[col] = np.log1p(data[col])

# #IQR
# for col in features_IQR:
#     Q1 = data[col].quantile(0.25)
#     Q3 = data[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     data[col]=data[col].clip(lower=lower_bound, upper=upper_bound)

#Scaling
scaling_data = data[features_scaling]
scaler = StandardScaler()
D = scaler.fit_transform(scaling_data)

#PCA
pca = PCA(n_components=2)
D_sample=D
data_pca = pca.fit_transform(D_sample)
D = pd.DataFrame(D, columns=features_scaling)

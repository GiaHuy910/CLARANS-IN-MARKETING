#đây mới là file để lấy data
import pandas as pd
import numpy as np 
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

path=r'../customer_clustering/datasets/Customer_Segmentation.csv'
data=pd.read_csv(path)

#tiền xử lí dữ liệu
data['Income']=data['Income'].fillna(data['Income'].median())

data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner",
                                                    "Together":"Partner", 
                                                    "Absurd":"Alone", 
                                                    "Widow":"Alone", 
                                                    "YOLO":"Alone", 
                                                    "Divorced":"Alone", 
                                                    "Single":"Alone",})

data['Children']=data['Kidhome']+data['Teenhome']

data["Family_Size"] = data["Living_With"].replace({"Alone": 1, 
                                        "Partner":2})+ data["Children"]

data["Education"]=data["Education"].replace({"Basic":"Undergraduate",
                                            "2n Cycle":"Undergraduate",
                                            "Graduation":"Graduate",
                                            "Master":"Postgraduate", 
                                            "PhD":"Postgraduate"})

data['Age']=datetime.date.today().year-data['Year_Birth']

data['Total_Spend']=data['MntWines']+data['MntFruits']+data['MntMeatProducts']+data['MntFishProducts']+data['MntSweetProducts']+data['MntGoldProds']

data=data.rename(columns={"MntWines": "Wines"
                        ,"MntFruits":"Fruits"
                        ,"MntMeatProducts":"Meat"
                        ,"MntFishProducts":"Fish"
                        ,"MntSweetProducts":"Sweets"
                        ,"MntGoldProds":"Gold"})

data['Food']=data['Meat']+data['Fish']+data['Sweets']+data['Fruits']
data['Wines_ratio']=(data['Wines']/data['Total_Spend']).round(3)
data['Food_ratio']=(data['Food']/data['Total_Spend']).round(3)
data['Gold_ratio']=(data['Gold']/data['Total_Spend']).round(3)

features_to_use=['Age','Income','Education'
                ,'Family_Size','Total_Spend'
                ,'Wines','Food','Gold'
                ,'Wines_ratio','Food_ratio','Gold_ratio'
                ,'NumWebPurchases'
                ,'NumDealsPurchases','NumCatalogPurchases'
                ,'NumStorePurchases','NumWebVisitsMonth']
data_to_use=data[features_to_use].copy()

#Số hoá kdl categorical
s = (data_to_use.dtypes == 'object')
object_cols = list(s[s].index)
LE=LabelEncoder()
for i in object_cols:
    data_to_use[i]=data_to_use[[i]].apply(LE.fit_transform)

#Cap
data_to_use = data_to_use[(data_to_use["Age"]<=90)]
data_to_use = data_to_use[(data_to_use["Income"]<150000)]

#Log
log_cols = ['Wines','Food','Gold','Total_Spend']
for col in log_cols:
    high=data[col].quantile(0.99)
    data_to_use[col]=data_to_use[col].clip(upper=high)

for col in log_cols:
    data_to_use[col]=np.log1p(data_to_use[col])

#Sau cùng chuẩn hoá dữ liệu số bằng standard scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use)
D = pd.DataFrame(scaled_data, columns=features_to_use)

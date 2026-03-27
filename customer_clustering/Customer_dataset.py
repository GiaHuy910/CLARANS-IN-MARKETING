import pandas as pd
path=r'../customer_clustering/datasets/customer_segmentation_data.csv'
data=pd.read_csv(path)
# data['Purchase_History'] = pd.to_datetime(
#     data['Purchase_History'],
#     format='%m/%d/%Y',
#     dayfirst=True
# )
# data['Date_num'] = (pd.Timestamp.today() - data['Purchase_History']).dt.days
# data=data.drop(columns=['Customer_ID','Segmentation_Group','Purchase_History'])
data=data.drop(columns=['Customer_ID','Segmentation_Group','Purchase_History','Marital_Status','Education_Level','Geographic_Information','Occupation','Behavioral_Data','Interactions_with_Customer_Service','Insurance_Products_Owned','Coverage_Amount','Premium_Amount','Policy_Type','Customer_Preferences','Preferred_Communication_Channel','Preferred_Contact_Time','Preferred_Language'])
data = data.dropna()
data=pd.get_dummies(data,dtype=int,drop_first=True)

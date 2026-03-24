import pandas as pd
data=pd.read_csv(r'ravalsmit/customer-segmentation-data/versions/1/customer_segmentation_data.csv')
Y=data.Segmentation_Group
data=data.drop(['Customer_ID','Segmentation_Group'],axis=1)
processed_data=pd.get_dummies(data)

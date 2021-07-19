import pandas as pd
import numpy as np
import sklearn
import pickle
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("C:/SUsers/hp/Desktop/python/Flask2/salary_predict_dataset.csv")
data['test_score'].fillna(0,inplace=True)
data['experience'].fillna('zero',inplace=True)
data['interview_score'].fillna(0,inplace=True)
def string_to_number(word):
    dict={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15}
    return dict[word]
data['experience']=data['experience'].apply(lambda x: string_to_number(x))
x = data.iloc[:,:3]
y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=5)
mymodel = LinearRegression()
mymodel.fit(x_train,y_train)
y_pred = mymodel.predict(x_test)
pickle.dump(mymodel,open("model.pkl","wb"))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))

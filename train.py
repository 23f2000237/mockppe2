import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df=pd.read_csv("data.csv")

X,y=df.iloc[:,:-1],df.iloc[:,-1]
model=LogisticRegression()
model.fit(X,y)

joblib.dump(model,"model.joblib")
print("Created joblib model")

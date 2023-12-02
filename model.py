#Importing libraries
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


data = load_iris()
print(dir(data))

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
print(df)

df = df.sample(frac=1)
print(df)

X = df.drop(columns="target").values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

with open("model.pkl", "wb") as file:
	pickle.dump(model, file)







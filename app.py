from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("collegePlace.csv")

value = df["Gender"].value_counts().keys()
value1 = df["Stream"].value_counts().keys()

for num, var in enumerate(value):
    df["Gender"].replace(var, num, inplace=True)

for num, var in enumerate(value1):
    df["Stream"].replace(var, num, inplace=True)

X = df.drop("PlacedOrNot", axis=1)
y = df["PlacedOrNot"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=123)

sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("placement_student_classifie.pkl")

def placement_student_classifie(model,Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs):
    for num,var in enumerate(value):
        if var == Gender:
            Gender = num
    for num1,var1 in enumerate(value1):
        if var1 == Stream:
            Stream = num1
            
            
    x = np.zeros(len(X.columns))
    x[0] = Age
    x[1] = Gender
    x[2] = Stream
    x[3] = Internships
    x[4] = CGPA
    x[5] = Hostel
    x[6] = HistoryOfBacklogs
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Age = request.form["Age"]
    Gender = request.form["Gender"]
    Stream = request.form["Stream"]
    Internships = request.form["Internships"]
    CGPA = request.form["CGPA"]
    Hostel = request.form["Hostel"]
    HistoryOfBacklogs = request.form["HistoryOfBacklogs"]
    
    predicated_price = placement_student_classifie(model,Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs)

    if predicated_price==1:
        return render_template("index.html", prediction_text="Student have placement is complate in any company.")
    else:
        return render_template("index.html", prediction_text="Student have don't complate placement..contact to collage!")


if __name__ == "__main__":
    app.run()    
    
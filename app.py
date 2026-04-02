from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("heart.csv")
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
cols = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
target_col = ["output"]

df = pd.get_dummies(df, columns = cat_cols, drop_first = True)
X = df.drop(['output'],axis=1)
y = df[['output']]
scaler = RobustScaler()
X[con_cols] = scaler.fit_transform(X[con_cols])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = np.argmax(y_pred_proba,axis=1)

app = Flask (__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    message = None
    style = "display: none;"
    if request.method == "POST":
        style="display: block"
        age = int(request.form.get("age"))
        trtbps = float(request.form.get("trtbps"))
        chol = float(request.form.get("chol"))
        thalachh = float(request.form.get("thalachh"))
        oldpeak = float(request.form.get("oldpeak"))
        sex = int(request.form.get("sex"))
        exng = int(request.form.get("exng"))
        caa = int(request.form.get("caa"))
        cp = int(request.form.get("cp"))
        fbs = int(request.form.get("fbs"))
        restecg = int(request.form.get("restecg"))
        slp = int(request.form.get("slp"))
        thall = int(request.form.get("thall"))

        user_input = {
            'age': age,
        	'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
        	'oldpeak': oldpeak, 
            'slp': slp,
            'caa': caa,
            'thall': thall
        }

        idf = pd.DataFrame([user_input], columns = cols)
        idf = pd.get_dummies(idf, columns=cat_cols, drop_first=True)
        idf = idf.reindex(columns=X.columns, fill_value=0)
        idf[con_cols] = scaler.fit_transform(idf[con_cols])
        prediction = model.predict(idf)
        if prediction[0] == 1:
            message = "Heart disease detected!"
        else:
            message = "Heart disease not detected."
        return render_template('home.html', message=message, style=style)
    return render_template('home.html', message=None, style=style)

if __name__ == '__main__':
    app.run(debug=True)
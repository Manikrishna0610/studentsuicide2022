#Backend code part for deployment - 'Natural' folder
#Implementing "Flask" concept via python

#pip install flask
#Importing 'Flask' package from 'flask' library
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

#Initializing the Flask web app..
app = Flask(__name__)
#from sklearn.externals import joblib #Serializing the model
import sklearn.externals
import joblib
import pickle
mmodel = pickle.load(open("simplinear.pkl",'rb'))


#@app.route('/test')
#def test():
#    return "Flask is used as a webservice for deployment process for our ML Model"

#Loading the model("simplinear.pkl.pkl") in this space(@starting itself) will increase the performance of the model and need not refresh again and again to load the model,just if in case.
#Here, the model will be loaded only once.

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/success", methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        data = pd.read_csv(f)
        data1 = data.iloc[:,2:].to_numpy()
        #Using the saved model..
        y2 = pd.DataFrame(mmodel.predict(data1),columns = ['SuicidalTendancy'])
        data['SuicidalTendancy'] = y2
        data.to_csv('results.csv',index = False)
        return render_template('data.html',Z="Your results are displayed.Kindly refer the generated results file in the CWD", Y=data.to_html())


if __name__ == "__main__":
    app.run(debug=True)

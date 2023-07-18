from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def main():
    return render_template("index.html")

def print_output(count0,count1,aws_df):
    total_ddos = count0[1][0]
    total_normal = count0[1][1]
    
    found_ddos = count1[1][0]
    found_normal = count1[1][1]
    
    ddos_per = (found_ddos/len(aws_df))*100
    normal_per = (found_normal/len(aws_df))*100
    
    out = [[0 for _ in range(2)]for _ in range(2)]
    out[0][0] = found_ddos
    out[0][1] = found_normal
    out[1][0] = ddos_per
    out[1][1] = normal_per
    return out

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))   
    df = df.drop(['start','end'],axis = 1) 

    le_SIP = preprocessing.LabelEncoder()
    le_SIP.fit(df['Source IP'])
    df['Source IP'] = le_SIP.fit_transform(df['Source IP'])

    le_DAD = preprocessing.LabelEncoder()
    le_DAD.fit(df['Destination IP'])
    df['Destination IP'] = le_DAD.fit_transform(df['Destination IP'])

    le_timestamp1 = preprocessing.LabelEncoder()
    le_timestamp1.fit(df['Timestamp'])
    df['Timestamp'] = le_timestamp1.fit_transform(df['Timestamp'])

    df = df.replace(np.inf, np.nan)
    df = df.dropna(axis=0)

    # load the model from disk
    loaded_model = pickle.load(open("model.pkl", 'rb'))
    aws_y_pred_vcs = loaded_model.predict(df)
    unique, frequency = np.unique(aws_y_pred_vcs, 
                              return_counts = True) 
    count0 = ([0,1],[1015392,2834])
    count1 = np.asarray((unique,frequency ))
    out = print_output(count0,count1,df)
    return render_template("result.html",result=out)

if __name__ == "__main__":
    app.run()
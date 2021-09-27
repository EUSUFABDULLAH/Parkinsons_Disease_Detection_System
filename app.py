from flask import Flask
from flask import Flask, render_template, request
import pickle
import sklearn
import numpy as np

try:
    model_KNN = pickle.load(open('parkinson_KNN.pkl', 'rb'))
    model_LG = pickle.load(open('parkinson_LG.pkl', 'rb'))
    model_RFC = pickle.load(open('parkinson_RFC.pkl', 'rb'))
    model_SVC = pickle.load(open('parkinson_SVC.pkl', 'rb'))
    model_XGBC = pickle.load(open('parkinson_XGBC.pkl', 'rb'))
except FileNotFoundError as e:
    print(e)


app = Flask(__name__)
@app.route("/", methods = ['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        mdvp_fo = float(request.form['MDVP:Fo(Hz)'])
        mdvp_fhi = float(request.form['MDVP:Fhi(Hz)'])
        mdvp_flo = float(request.form['MDVP:Flo(Hz)'])
        mdvp_jitter_per = float(request.form['MDVP:Jitter(%)'])
        mdvp_jitter_abs = float(request.form['MDVP:Jitter(Abs)'])
        mdvp_rap = float(request.form['MDVP:RAP'])
        mdvp_ppq = float(request.form['MDVP:PPQ'])
        jitter_ddp = float(request.form['Jitter:DDP'])
        mdvp_shimmer = float(request.form['MDVP:Shimmer'])
        mdvp_shimmer_db = float(request.form['MDVP:Shimmer(dB)'])
        shimmer_apq3 = float(request.form['Shimmer:APQ3'])
        shimmer_apq5 = float(request.form['Shimmer:APQ5'])
        mdvp_apq = float(request.form['MDVP:APQ'])
        shimmer_dda = float(request.form['Shimmer:DDA'])
        nhr = float(request.form['NHR'])
        hnr = float(request.form['HNR'])
        rpde = float(request.form['RPDE'])
        dfa = float(request.form['DFA'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        d2 = float(request.form['D2'])
        ppe = float(request.form['PPE'])
        algo = int(request.form['algo'])

        if algo == 1:
            output = model_XGBC.predict(np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_per, mdvp_jitter_abs, mdvp_rap,
                                                   mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                                                   shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                                                   spread2, d2, ppe]]))
        elif algo == 2:
            output = model_RFC.predict(np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_per, mdvp_jitter_abs, mdvp_rap,
                                              mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                                              shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                                              spread2, d2, ppe]]))
        elif algo == 3:
            output = model_LG.predict(np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_per, mdvp_jitter_abs, mdvp_rap,
                                              mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                                              shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                                              spread2, d2, ppe]]))
        elif algo == 4:
            output = model_KNN.predict(np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_per, mdvp_jitter_abs, mdvp_rap,
                                              mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                                              shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                                              spread2, d2, ppe]]))
        elif algo == 5:
            output = model_SVC.predict(np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_per, mdvp_jitter_abs, mdvp_rap,
                                              mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3,
                                              shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                                              spread2, d2, ppe]]))

        res = ""
        if output == 1:
            res = "in risk"
            return render_template('Prediction.html', predict_res=res)
        else:
            res = "risk free"
            return render_template('Prediction_RF.html',predict_res = res)

@app.route("/result", methods=['GET'])
def result():
    pass


if __name__ == "__main__":
    app.debug = True
    app.run()
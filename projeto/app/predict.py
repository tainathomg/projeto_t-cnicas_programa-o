# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
import joblib
from bottle import route, run


@route('/', method='POST')
def send():
    #Tratando os dados de entrada
    df_predict = pd.read_csv("input/predict.csv",  delimiter=';')
    X_predict = df_predict.drop(['Amostra'], axis='columns')

    #Abrindo modelo e fazendo previsão
    svm = joblib.load('model.pkl')
    Y_svm = svm.predict(X_predict)
    
    #Tabela com resultado predito
    Y_predito = pd.DataFrame(data = Y_svm, columns = ['Y_pred_svm'], index = X_predict.index.copy())
    Y_predito_svm = pd.merge(df_predict['Amostra'], Y_predito, how = 'left', left_index = True, right_index = True)
    Y_predito_svm.to_csv('output/output_predict.csv', sep=';')
   
    return 'A previsão dos seus dados foi processada! O arquivo referente ao resultado da classificação encontra-se na pasta de trabalho como output_predict.csv.'


if __name__ == '__main__':
	run(host='0.0.0.0', port=8081, debug=True)    
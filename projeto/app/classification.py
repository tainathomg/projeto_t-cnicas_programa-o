# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import joblib
from bottle import route, run


def svm(X_train, Y_train, X_test, Y_test, df):
    svm = SVC(gamma='scale', kernel='linear', C=10)
    svm.fit(X_train, Y_train)
    Y_pred_svm = svm.predict(X_test)
    
    # Output a pickle file for the model
    joblib.dump(svm, 'model.pkl') 
    print ('Arquivo com modelo de classificação gerado exportado com sucesso!')
    print()
    
    #Tabela com comparativo entre esperado e predito (conjunto de teste)
    Y_pred = pd.DataFrame(data = Y_pred_svm, columns = ['Y_pred_svm'], index = Y_test.index.copy())
    Y_comp = pd.merge(Y_test, Y_pred, how = 'left', left_index = True, right_index = True)
    Y_comp = pd.merge(Y_comp, df['Amostra'], how = 'left', left_index = True, right_index = True)
    Y_comp.to_csv('output/output_test_classification.csv', sep=';')
    print ('Arquivo com saída do conjunto de teste da classificação (.csv) exportado com sucesso!')
    print()
    
    #Matriz de confusão
    matriz_svm = pd.crosstab(Y_test,Y_pred_svm, rownames=['Real'], colnames=['Predito'], margins=True)
    print ('Matriz de Confusão:')
    print (matriz_svm)
    matriz_svm.to_csv('output/matriz.csv', sep=';')
    print ('Arquivo com matriz de confusão (.csv) exportado com sucesso!')
    print()
    
    #Métricas de desempenho
    metrics_svm = metrics.classification_report(Y_test,Y_pred_svm, output_dict=True)
    metrics_svm = pd.DataFrame(metrics_svm).transpose()
    print ('Métricas:')
    print(metrics_svm)
    metrics_svm.to_csv('output/metrics.csv', sep=';')
    print ('Arquivo com métricas (.csv) exportado com sucesso!')
    print()
    
    return print('Classificação dos dados por algoritmo SVM concluída com sucesso!')


@route('/', method='POST')
def send():
    #Tratando os dados de entrada
    df = pd.read_csv("input/classification.csv",  delimiter=';')
    X = df.drop(['Classe', 'Amostra'], axis='columns')
    Y = df.Classe

    #Conjunto de treino e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
    
    #Classificação
    svm(X_train, Y_train, X_test, Y_test, df)
        
    return 'A classificação dos seus dados foi processada e um modelo foi gerado! Os arquivos referentes aos resultados da classificação, matriz de confusão e métricas encontram-se na pasta de trabalho "app/output".'


if __name__ == '__main__':
	run(host='0.0.0.0', port=8080, debug=True)    
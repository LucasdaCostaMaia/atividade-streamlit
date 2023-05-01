import time
import numpy as np
import pandas as pd
import streamlit as st
import sklearn.metrics
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

st.write("*Ciência da Computação*")
st.write("*Deploy de Modelos de Machine Learning*")
st.write("*Deploy de Aplicações Preditivas com Streamlit*")
st.title("Aplicações Preditivas usando a Regressão Logística")

st.sidebar.header('Dataset e Hiperparâmetros')
st.sidebar.markdown("""**Selecione o Dataset Desejado**""")
Dataset = st.sidebar.selectbox('Dataset', ('Iris', 'Wine', 'Breast Cancer'))
Split = st.sidebar.slider('Escolha o Percentual de Divisão dos Dados em Treino e Teste (padrão = 70/30):', 0.1, 0.9,
                          0.70)
st.sidebar.markdown("""**Selecione os Hiperparâmetros Para o Modelo de Regressão Logística**""")
Solver = st.sidebar.selectbox('Algoritmo', ('lbfgs', 'newton-cg', 'liblinear', 'sag'))
Penality = st.sidebar.radio("Regularização:", ('none', 'l1', 'l2', 'elasticnet'))
Tol = st.sidebar.text_input("Tolerância Para Critério de Parada (default = 1e-4):", "1e-4")
Max_Iteration = st.sidebar.text_input("Número de Iterações (default = 50):", "50")

parameters = {'Penality': Penality, 'Tol': Tol, 'Max_Iteration': Max_Iteration, 'Solver': Solver}

def carrega_dataset(dataset):
    if dataset == 'Iris':
        dados = sklearn.datasets.load_iris()
    elif dataset == 'Wine':
        dados = sklearn.datasets.load_wine()
    elif dataset == 'Breast Cancer':
        dados = sklearn.datasets.load_breast_cancer()

    return dados


def prepara_dados(dados, split):
    
    X_treino, X_teste, y_treino, y_teste = train_test_split(dados.data, dados.target, test_size=float(split),
                                                            random_state=42)

    scaler = MinMaxScaler()

    X_treino = scaler.fit_transform(X_treino)

    X_teste = scaler.transform(X_teste)

    return (X_treino, X_teste, y_treino, y_teste)

def cria_modelo(parameters):
    X_treino, X_teste, y_treino, y_teste = prepara_dados(Data, Split)

    clf = LogisticRegression(penalty=parameters['Penality'],
                             solver=parameters['Solver'],
                             max_iter=int(parameters['Max_Iteration']),
                             tol=float(parameters['Tol']))

    clf = clf.fit(X_treino, y_treino)

    prediction = clf.predict(X_teste)

    accuracy = sklearn.metrics.accuracy_score(y_teste, prediction)

    cm = confusion_matrix(y_teste, prediction)

    dict_value = {"modelo": clf, "acuracia": accuracy, "previsao": prediction, "y_real": y_teste, "Metricas": cm,
                  "X_teste": X_teste}

    return (dict_value)



st.markdown("""Resumo dos Dados""")
st.write("Nome do Dataset:", Dataset)

Data = carrega_dataset(Dataset)

targets = Data.target_names

Dataframe = pd.DataFrame(Data.data, columns=Data.feature_names)
Dataframe['target'] = pd.Series(Data.target)
Dataframe['target labels'] = pd.Series(targets[i] for i in Data.target)

st.write("Visão Geral dos Atributos:")
st.write(Dataframe)


if (st.sidebar.button("Clique Para Treinar o Modelo de Regressão Logística")):

    with st.spinner('Carregando o Dataset...'):
        time.sleep(1)

    st.success("Dataset Carregado!")

    modelo = cria_modelo(parameters)

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    with st.spinner('Treinando o Modelo...'):
        time.sleep(1)

    st.success("Modelo Treinado")

    labels_reais = [targets[i] for i in modelo["y_real"]]

    labels_previstos = [targets[i] for i in modelo["previsao"]]

    st.subheader("Previsões do Modelo nos Dados de Teste")

    st.write(pd.DataFrame({"Valor Real": modelo["y_real"],
                           "Label Real": labels_reais,
                           "Valor Previsto": modelo["previsao"],
                           "Label Previsto": labels_previstos, }))

    matriz = modelo["Metricas"]

    st.subheader("Matriz de Confusão nos Dados de Teste")

    st.write(matriz)

    st.write("Acurácia do Modelo:", modelo["acuracia"])

    st.write("Obrigado por usar esta app do Streamlit!")


import os
import random

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def reset_seeds():
    """
    Define sementes fixas para tornar os resultados reproduzíveis.

    Ajusta as sementes de `os`, `tensorflow`, `numpy` e `random` para que
    execuções diferentes usem a mesma sequência pseudoaleatória.

    Parâmetros:
        None

    Retorno:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def read_data():
    """
    Lê o dataset de cardiotocografia e separa variáveis de entrada e alvo.

    Retorno:
        tuple[pandas.DataFrame, pandas.Series]: `X` com as variáveis preditoras
        e `y` com a coluna alvo `fetal_health`.
    """

    url = 'raw.githubusercontent.com'
    username = 'renansantosmendes'
    repository = 'lectures-cdas-2023'
    file_name = 'fetal_health_reduced.csv'
    data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')

    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return X, y


def process_data(X, y):
    """
    Padroniza as variáveis e divide os dados em treino e teste.

    Parâmetros:
        X (pandas.DataFrame): Variáveis de entrada.
        y (pandas.Series): Rótulos originais da classe fetal.

    Retorno:
        X_train (pandas.DataFrame): Dados de treino padronizados.
        X_test (pandas.DataFrame): Dados de teste padronizados.
        y_train (pandas.Series): Rótulos de treino ajustados para iniciar em 0.
        y_test (pandas.Series): Rótulos de teste ajustados para iniciar em 0.
    """
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    y_train = y_train - 1
    y_test = y_test - 1
    return X_train, X_test, y_train, y_test


def create_model(X):
    """
    Cria e compila uma rede neural densa para classificação multiclasse.

    Parâmetros:
        X (numpy.ndarray): Dados de entrada com formato
        `(num_amostras, num_variaveis)`.

    Retorno:
        tensorflow.keras.models.Sequential: Modelo compilado pronto para treino.
    """
    reset_seeds()
    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def config_mlflow():
    """
    Configura o rastreamento de experimentos no MLflow.

    Define variáveis de ambiente de autenticação, configura a URI de tracking
    e habilita o autolog do Keras para registrar parâmetros, métricas e artefatos.

    Observação:
        A URI efetivamente usada é a definida em `mlflow.set_tracking_uri(...)`.

    Parâmetros:
        None

    Retorno:
        None
    """
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'foliveiradiass'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c4cda36a78a9721ef466b66e9a3cfbbabf820900'
    mlflow.set_tracking_uri('https://dagshub.com/foliveiradiass/mlops-ead.mlflow')

    mlflow.keras.autolog(log_models=True,
                         log_input_examples=True,
                         log_model_signatures=True)


def train_model(model, X_train, y_train, is_train=True):
    """
    Treina o modelo com MLflow ativo e registra o modelo opcionalmente.

    Parâmetros:
    - model: Modelo Keras já compilado.
    - X_train: Dados de treino.
    - y_train: Rótulos de treino.
    - is_train: Se `True`, registra o modelo no Model Registry do MLflow.

    Retorno:
        None
    """
    with mlflow.start_run(run_name='experiment_mlops_ead') as run:
        model.fit(X_train,
                  y_train,
                  epochs=50,
                  validation_split=0.2,
                  verbose=3)
    if is_train:
        run_uri = f'runs:/{run.info.run_id}'
        mlflow.register_model(run_uri, 'fetal_health')


def main():
    """
    Executa o pipeline completo: carrega dados, processa, cria modelo,
    configura MLflow e treina.
    """
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X)
    config_mlflow()
    train_model(model, X_train, y_train)


if __name__ == "__main__":
    main()


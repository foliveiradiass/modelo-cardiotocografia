import os
import pandas as pd
import pytest
import numpy as np
from tensorflow.keras.models import Sequential

from train import (read_data,
                   create_model,
                   train_model,
                   process_data,
                   reset_seeds,
                   config_mlflow,
                   main)


@pytest.fixture
def sample_data():
    """
    Fixture que retorna um conjunto de dados de exemplo.

    Retorno:
        pandas.DataFrame: DataFrame com 'feature1', 'feature2' e 'fetal_health'.
    """
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'fetal_health': [1, 1, 2, 3, 2]
    })
    return data


def test_reset_seeds():
    """
    Testa se reset_seeds define as sementes corretamente para reprodutibilidade.
    """
    reset_seeds()
    val1 = np.random.rand()
    
    reset_seeds()
    val2 = np.random.rand()
    
    assert val1 == val2, "Sementes não foram resetadas corretamente"


def test_read_data():
    """
    Testa se read_data retorna dados não vazios para X e y.
    """
    X, y = read_data()

    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]


def test_process_data(sample_data):
    """
    Testa se process_data padroniza e divide os dados corretamente.
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health']
    
    X_train, X_test, y_train, y_test = process_data(X, y)
    
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert X_train.shape[1] == X.shape[1]
    assert (y_train.min() >= 0) and (y_train.max() <= 2)
    assert (y_test.min() >= 0) and (y_test.max() <= 2)


def test_create_model(sample_data):
    """
    Testa se create_model retorna um modelo Keras compilado corretamente.
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    model = create_model(X)

    assert len(model.layers) > 2
    assert model.trainable
    assert isinstance(model, Sequential)
    assert model.optimizer is not None


def test_train_model(sample_data):
    """
    Testa se train_model treina o modelo sem erros (is_train=False).
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health'] - 1
    model = create_model(X)
    train_model(model, X, y, is_train=False)
    assert model.history.history['loss'][-1] > 0
    assert model.history.history['val_loss'][-1] > 0


def test_train_model_with_registration(sample_data):
    """
    Testa se train_model registra o modelo quando is_train=True.
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health'] - 1
    model = create_model(X)
    config_mlflow()
    train_model(model, X, y, is_train=True)
    assert model.history.history['loss'][-1] > 0
    assert model.history.history['val_loss'][-1] > 0


def test_config_mlflow():
    """
    Testa se config_mlflow define as variáveis de ambiente corretamente.
    """
    config_mlflow()
    
    assert os.environ.get('MLFLOW_TRACKING_USERNAME') is not None
    assert os.environ.get('MLFLOW_TRACKING_PASSWORD') is not None


def test_main_execution():
    """
    Testa o fluxo principal: read_data -> process_data -> create_model -> train_model.
    """
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X)
    config_mlflow()
    train_model(model, X_train, y_train)
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert model.history.history['loss'][-1] > 0


def test_main_function():
    """
    Testa a função main que executa todo o pipeline.
    """
    main()
    assert True  # Se chegou aqui sem exceção, o main() executou com sucesso



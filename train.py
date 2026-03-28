

"""# 1 - Importando os módulos necessários"""

import os
import random
import mlflow
import dagshub
import numpy as np
import random as python_random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""# Definindo funções adicionais"""

def reset_seeds() -> None:
  """
  Resets the seeds to ensure reproducibility of results.

  This function sets the seed for various random number generation libraries
  to ensure that results are reproducible. The affected libraries are:
  - Python's built-in `random`
  - NumPy
  - TensorFlow

  The seed used is 42.

  Returns:
      None
  """
  os.environ['PYTHONHASHSEED']=str(42)
  tf.random.set_seed(42)
  np.random.seed(42)
  random.seed(42)

"""# 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis"""

url = 'raw.githubusercontent.com'
username = 'renansantosmendes'
repository = 'lectures-cdas-2023'
file_name = 'fetal_health_reduced.csv'
data = pd.read_csv(f'https://{url}/{username}/{repository}/master/{file_name}')


"""# 3 - Preparando o dado antes de iniciar o treino do modelo"""

X=data.drop(["fetal_health"], axis=1)
y=data["fetal_health"]

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

y_train = y_train -1
y_test = y_test - 1

"""# 4 - Criando o modelo e adicionando as camadas"""

reset_seeds()
model = Sequential()
model.add(InputLayer(shape=(X_train.shape[1], )))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

"""# 5 - Compilando o modelo

"""

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

"""##**Configurando o mlflow**"""



os.environ['MLFLOW_TRACKING_USERNAME'] = 'foliveiradiass'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c4cda36a78a9721ef466b66e9a3cfbbabf820900'
mlflow.set_tracking_uri('https://dagshub.com/foliveiradiass/mlops-ead.mlflow')

mlflow.keras.autolog(log_models=True,
                     log_input_examples=True,
                     log_model_signatures=True)

dagshub.init(
    repo_name="mlops-ead",      # nome do repo
    repo_owner="foliveiradiass",   # seu username no DagsHub
    mlflow=True
)

"""# 6 - Executando o treino do modelo"""

with mlflow.start_run(run_name='experiment_mlops_ead') as run:
  model.fit(X_train,
            y_train,
            epochs=50,
            validation_split=0.2,
            verbose=3)


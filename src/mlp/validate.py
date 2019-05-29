import os
import warnings

# Filtro para warnings irritantes
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy
import keras
import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Definição fixa das sementes dos geradores aleatórios
# para facilitar a reprodução dos resultados
numpy.random.seed(1)
tensorflow.set_random_seed(1)

samples = list()

# Leitura da base de dados de segmentação de pele
# Cada linha contém uma amostra
# Os três primeiros números são os valores B, G e R de um pixel
# O quarto número corresponde à classe do pixel: 1 para pele, 2 para não-pele
# 50 859 amostras de pele e 194 198 de não-pele
# Link: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
with open('skin.txt') as skin:
    for row in skin.readlines():
        samples.append(list(map(int, row.split())))
        samples[-1][-1] += -1

# Representação do dataset como um array numpy
dataset = numpy.array(samples)
# Realiza uma permutação aleatória das amostras
numpy.random.shuffle(dataset)
# Trucamento do dataset para 5% das amostras (12 252 amostras)
dataset = dataset[:int(0.05*len(dataset)),:]

# Número de amostras e dimensões dos vetores de entrada
n, d = dataset.shape[0], dataset.shape[1] - 1

# Preenchimento do vetor de amostras X e suas classes Y
X = dataset[:,:d]
Y = dataset[:,d:]

# Instanciação do objeto responsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia K-Fold
cross_val = StratifiedKFold(5)
cross_val.get_n_splits(X)

# Variável para armazenar acurácias
kfold_scores = list()

# Percorre as divisões de conjuntos de treino e teste K-Fold
for train_index, test_index in cross_val.split(X,Y):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Alteração da codificação do vetor de classes para one-hot-encoding
    # 0 torna-se [1, 0] e 1 torna-se [0, 1]
    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = keras.utils.to_categorical(Y_test)

    # Instanciação de um modelo sequencial;
    # Este modelo é uma pilha de camadas de neurônios;
    # Sua construção é feita através da adição sequencial de camadas,
    # primeiramente a camada de entrada, depois as camadas e ocultas e, 
    # enfim, a camada de saída;
    # Neste exemplo, a classe Dense representa camadas totalmente conectadas
    model = keras.models.Sequential([
        keras.layers.Dense(6, activation='sigmoid', input_shape=(d,)),
        keras.layers.Dense(2, activation='softmax')
    ])

    # Compilação do modelo
    # Definição do algoritmo de otimização e da função de perda
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Critério de parada
    # Caso a perda sobre a validação não decresça por 2 épocas,
    # o treinamento é interrompido
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    # Treinamento
    # Executa o algoritmo de otimização, ajustando os pesos das conexões
    # da rede neural com base nos valores de entrada X e saída Y, usando
    # a função de perda como forma de verificar o quão corretas são suas
    # predições durante o treinamento. Realiza no máximo 20 passagens pelo 
    # conjunto de treinamento. Utiliza 20% dos conjuntos X e Y como validação.
    history = model.fit(X_train, Y_train, epochs=20,
                        validation_split=0.2, callbacks=[stop], verbose=0)
    
    # Teste
    # Avalia a rede neural treinada sobre o conjunto de teste e calcula a acurácia
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f %%" % (model.metrics_names[1], scores[1]*100))

    # Armazena a acurácia
    kfold_scores.append(scores[1]*100)

# Exibe a média dos resultados obtidos sobre todos os folds, junto com o desvio
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(kfold_scores), numpy.std(kfold_scores)))
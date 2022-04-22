# Esse código foi feito a base de um tutorial.
# Realizando as importações necessárias:
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Modelo da rede neural é sequencial, ou seja, camada atrás de camada.
        model = Sequential()
        # Adicionando as informações das imagens: altura, largura e profundidade.
        inputShape = (height, width, depth)
        chanDim = -1

        # Análise do back-end: o TensorFlow utiliza o modelo "channels_last".
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # O primeiro parâmetro indica a quantidade de filtros a ser usado na camada.
        # Quantidade de filtros resulta na quantidade de "feature maps".
        # O segundo parâmetro indica o tamanho do "kernel", que sempre vem em números ímpares.
        # O terceiro parâmetro é o preenchimento por zero: não perde muita informação da imagem.
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        # Função de ativação: trazer as não linearidades ao sistema.
        model.add(Activation("relu"))
        # Normalização do lote:
        model.add(BatchNormalization(axis=chanDim))
        # Uso do "pooling": diminuir a dimensão espacial do volume de saída.
        model.add(MaxPooling2D(pool_size=(3, 3)))
        # Uso do "dropout": zerar determinada procentagem dos neurônios para evitar o "overfitting".
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # Adição de mais uma camada de convolução para ser reconhecido mais características das imagens.
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Adição do "flatenning": transformação das informações em duas dimenesões para somente uma.
        model.add(Flatten())
        # Adição da rede neural densa.
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Adição da camada de saída: a quantidade de neurônios será igual a  quantidade de classes utilizadas.
        model.add(Dense(classes))
        # Adição da função de ativação "softmax": utiliza o maior valor das probabilidades.
        model.add(Activation("softmax"))

        return model

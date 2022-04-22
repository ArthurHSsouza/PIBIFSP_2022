# Comando para treinar a rede via terminal:
# python train.py --dataset shapes/dataset --model flowchart2.model --labelbin lb2.pickle

# Esse código foi feito por meio de um tutorial.
# Importações de bibliotecas necessárias:
import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Envio dos argumetnos para a execução desse script pelo terminal.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Diretório com o dataset")
ap.add_argument("-m", "--model", required=True,
                help="Nome do arquivo .model gerado")
ap.add_argument("-l", "--labelbin", required=True,
                help="Nome do arquivo .pickle gerado")
args = vars(ap.parse_args())

# Quantidadede épocas: quantas vezes serão ajustados os pesos para a diminuição do erro.
EPOCHS = 100
# Taxa de aprendizagem.
INIT_LR = 1e-3
# Tamanho do lote que indica a quantidade de registros para a atualização dos pesos.
BS = 32
# As dimensões utilizadas para as imagens: altura, largura e cor/profundidade.
IMAGE_DIMS = (96, 96, 3)

# Inicializando os dados e os rótulos deles.
data = []
labels = []

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Alterar o tipo da varíavel dos dados, pois o tipo "float" possui melhor suporte.
# Aplicação da normalização dos valores de pixels para acelerar o processamento: colocá-los em uma escalada de 0 e 1 dividindo-os pela escala RGB.
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Divisão da base de dados para teste e treinamento.
# Uso do 'X' para os atributos previsores e do 'Y' para os atributos de classe.
# O parâmetro "test_size" indica a porcentagem destinada ao treinamento.
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# Uso do "data augumentation": aumento das variações dos dados existentes.
aug = ImageDataGenerator(rotation_range=0, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
# Criando o modelo da rede neural: envia-se parâemtros como as dimensões das imagens e suas classes.
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(lb.classes_))
# Configuração do otimizador: forma de encontrar o ponto mínimo de erro.
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# Configurando a função de perda e a métrica dos valores.
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# Somente imagens de treinamento tem sua quantidade aumentada.
# O "step_per_epoch" indica a quantidade de etapas que serão feitas por época, ou seja,
# a quantidade de conjuntos de imagens (lotes) que terão seus pesos reajustados.
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch= len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# Salvando a rede neural em formato ".model".
print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

scores = model.evaluate(testX, testY, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


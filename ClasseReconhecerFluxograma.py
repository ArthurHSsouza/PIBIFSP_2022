# Realizando as importações necessárias:
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pickle
import cv2
import numpy as np
import io
import json
import requests
import time

# Global.
LINMAX = 0
COLMAX = 2
inicioLin = 0
inicioCol = 0
fimLin = 0
fimCol = 0
fimCondicao = False
lin = 0
col = 0
variavelTipo = ""

# Matrizes.
quadranteLabel = []
quadranteTexto = []
quadranteLinOrigem = []
quadranteColOrigem = []
quadranteLinDestino = []
quadranteColDestino = []
quadranteLinDestino2 = []
quadranteColDestino2 = []


# Criando a classe de reconhecimento do fluxograma:
class ReconhecerFluxograma:

    def __init__(self):
        pass

    # Convertendo expressões para seus respectivos caracteres especiais.
    def editarTexto(self, textoOCR, linguagem):
        # Global.
        global variavelTipo

        listaTexto = textoOCR.split(" ")
        textoFinal = textoOCR

        for i in range(len(listaTexto)):
            if listaTexto[i] == 'IGUAL':
                if listaTexto[i - 1] == "MAIOR" or listaTexto[i - 1] == "MENOR":
                    textoFinal = textoFinal.replace("IGUAL", "=")
                else:
                    if linguagem == 1:
                        textoFinal = textoFinal.replace("IGUAL", "=")
                    elif linguagem == 2:
                        textoFinal = textoFinal.replace("IGUAL", "==")

            elif listaTexto[i] == 'MODULO':
                textoFinal = textoFinal.replace("MODULO", "%")

            elif listaTexto[i] == 'RECEBE':
                if linguagem == 1:
                    textoFinal = textoFinal.replace("RECEBE", "<-")
                elif linguagem == 2:
                    textoFinal = textoFinal.replace("RECEBE", "=")

            elif listaTexto[i] == 'TIPO':
                if linguagem == 1:
                    textoFinal = textoFinal.replace("TIPO", ":")
                    textoFinal = textoFinal.replace(listaTexto[i+1], listaTexto[i+1].lower())
                elif linguagem == 2:
                    variavelTipo = listaTexto[i+1].lower()
                    textoFinal = "#Variável do tipo {}".format(listaTexto[i+1].lower())

            elif listaTexto[i] == 'MAIOR':
                textoFinal = textoFinal.replace("MAIOR", ">")

            elif listaTexto[i] == 'MENOR':
                textoFinal = textoFinal.replace("MENOR", "<")

            elif listaTexto[i] == 'VEZES':
                textoFinal = textoFinal.replace("VEZES", "*")

            elif listaTexto[i] == 'DIVIDE':
                textoFinal = textoFinal.replace("DIVIDE", "/")

        return textoFinal

    # Realizando o recorte da área de interesse de maneira manual.
    def selecionarROI(self, imagem):
        # Redimensionando o tamanho da janela para não afetar o tamanho da imagem.
        cv2.namedWindow("Selecionar:", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Selecionar:', 500, 700)

        # Abrindo a janela de seleção ROI e realizando o recorte da caixa seletora.
        print("[INFO] Desenhe um retângulo na área que compõe o fluxograma.")
        meuROI = cv2.selectROI("Selecionar:", imagem, False, False)
        imagemCortada = imagem[int(meuROI[1]):int(meuROI[1] + meuROI[3]), int(meuROI[0]):int(meuROI[0] + meuROI[2])]
        cv2.destroyAllWindows()

        return imagemCortada

    # Recortando os quadrantes do painel, salvando-os no disco para o reconhecer o formato e o texto neles presente.
    def recortarQuadrantes1(self, imagemRedimensionada):
        # Linha 1
        cv2.imwrite("Quadrantes/0x0.jpg", imagemRedimensionada[0:200, 0:350])
        cv2.imwrite("Quadrantes/0x1.jpg", imagemRedimensionada[0:200, 350:680])
        cv2.imwrite("Quadrantes/0x2.jpg", imagemRedimensionada[0:200, 680:1000])

        # Linha 2
        cv2.imwrite("Quadrantes/1x0.jpg", imagemRedimensionada[200:370, 0:350])
        cv2.imwrite("Quadrantes/1x1.jpg", imagemRedimensionada[200:370, 350:680])
        cv2.imwrite("Quadrantes/1x2.jpg", imagemRedimensionada[200:370, 680:1000])

        # Linha 3
        cv2.imwrite("Quadrantes/2x0.jpg", imagemRedimensionada[370:530, 0:350])
        cv2.imwrite("Quadrantes/2x1.jpg", imagemRedimensionada[370:530, 350:680])
        cv2.imwrite("Quadrantes/2x2.jpg", imagemRedimensionada[370:530, 680:1000])

        # Linha 4
        cv2.imwrite("Quadrantes/3x0.jpg", imagemRedimensionada[530:710, 0:350])
        cv2.imwrite("Quadrantes/3x1.jpg", imagemRedimensionada[530:710, 350:680])
        cv2.imwrite("Quadrantes/3x2.jpg", imagemRedimensionada[530:710, 680:1000])

        # Linha 5
        cv2.imwrite("Quadrantes/4x0.jpg", imagemRedimensionada[710:880, 0:350])
        cv2.imwrite("Quadrantes/4x1.jpg", imagemRedimensionada[710:880, 350:680])
        cv2.imwrite("Quadrantes/4x2.jpg", imagemRedimensionada[710:880, 680:1000])

        # Linha 6
        cv2.imwrite("Quadrantes/5x0.jpg", imagemRedimensionada[880:1050, 0:350])
        cv2.imwrite("Quadrantes/5x1.jpg", imagemRedimensionada[880:1050, 350:680])
        cv2.imwrite("Quadrantes/5x2.jpg", imagemRedimensionada[880:1050, 680:1000])

        # Linha 7
        cv2.imwrite("Quadrantes/6x0.jpg", imagemRedimensionada[1050:1230, 0:350])
        cv2.imwrite("Quadrantes/6x1.jpg", imagemRedimensionada[1050:1230, 350:680])
        cv2.imwrite("Quadrantes/6x2.jpg", imagemRedimensionada[1050:1230, 680:1000])

        # Linha 8
        cv2.imwrite("Quadrantes/7x0.jpg", imagemRedimensionada[1230:1400, 0:350])
        cv2.imwrite("Quadrantes/7x1.jpg", imagemRedimensionada[1230:1400, 350:680])
        cv2.imwrite("Quadrantes/7x2.jpg", imagemRedimensionada[1230:1400, 680:1000])
    def recortarQuadrantes2(self, imagemRedimensionada):
        # Linha 1
        cv2.imwrite("Quadrantes/0x0.jpg", imagemRedimensionada[0:200, 0:350])
        cv2.imwrite("Quadrantes/0x1.jpg", imagemRedimensionada[0:200, 350:680])
        cv2.imwrite("Quadrantes/0x2.jpg", imagemRedimensionada[0:200, 680:1000])

        # Linha 2
        cv2.imwrite("Quadrantes/1x0.jpg", imagemRedimensionada[200:370, 0:350])
        cv2.imwrite("Quadrantes/1x1.jpg", imagemRedimensionada[200:370, 350:680])
        cv2.imwrite("Quadrantes/1x2.jpg", imagemRedimensionada[200:370, 680:1000])

        # Linha 3
        cv2.imwrite("Quadrantes/2x0.jpg", imagemRedimensionada[370:530, 0:350])
        cv2.imwrite("Quadrantes/2x1.jpg", imagemRedimensionada[370:530, 350:680])
        cv2.imwrite("Quadrantes/2x2.jpg", imagemRedimensionada[370:530, 680:1000])

        # Linha 4
        cv2.imwrite("Quadrantes/3x0.jpg", imagemRedimensionada[530:710, 0:350])
        cv2.imwrite("Quadrantes/3x1.jpg", imagemRedimensionada[530:710, 350:680])
        cv2.imwrite("Quadrantes/3x2.jpg", imagemRedimensionada[530:710, 680:1000])

        # Linha 5
        cv2.imwrite("Quadrantes/4x0.jpg", imagemRedimensionada[710:880, 0:350])
        cv2.imwrite("Quadrantes/4x1.jpg", imagemRedimensionada[710:880, 350:680])
        cv2.imwrite("Quadrantes/4x2.jpg", imagemRedimensionada[710:880, 680:1000])

        # Linha 6
        cv2.imwrite("Quadrantes/5x0.jpg", imagemRedimensionada[880:1050, 0:350])
        cv2.imwrite("Quadrantes/5x1.jpg", imagemRedimensionada[880:1050, 350:680])
        cv2.imwrite("Quadrantes/5x2.jpg", imagemRedimensionada[880:1050, 680:1000])

        # Linha 7
        cv2.imwrite("Quadrantes/6x0.jpg", imagemRedimensionada[1050:1230, 0:350])
        cv2.imwrite("Quadrantes/6x1.jpg", imagemRedimensionada[1050:1230, 350:680])
        cv2.imwrite("Quadrantes/6x2.jpg", imagemRedimensionada[1050:1230, 680:1000])

        # Linha 8
        cv2.imwrite("Quadrantes/7x0.jpg", imagemRedimensionada[1230:1400, 0:350])
        cv2.imwrite("Quadrantes/7x1.jpg", imagemRedimensionada[1230:1400, 350:680])
        cv2.imwrite("Quadrantes/7x2.jpg", imagemRedimensionada[1230:1400, 680:1000])

        # Linha 9
        cv2.imwrite("Quadrantes/8x0.jpg", imagemRedimensionada[1400:1580, 0:350])
        cv2.imwrite("Quadrantes/8x1.jpg", imagemRedimensionada[1400:1580, 350:680])
        cv2.imwrite("Quadrantes/8x2.jpg", imagemRedimensionada[1400:1580, 680:1000])

        # Linha 10
        cv2.imwrite("Quadrantes/9x0.jpg", imagemRedimensionada[1580:1760, 0:350])
        cv2.imwrite("Quadrantes/9x1.jpg", imagemRedimensionada[1580:1760, 350:680])
        cv2.imwrite("Quadrantes/9x2.jpg", imagemRedimensionada[1580:1760, 680:1000])

        # Linha 11
        cv2.imwrite("Quadrantes/10x0.jpg", imagemRedimensionada[1760:1930, 0:350])
        cv2.imwrite("Quadrantes/10x1.jpg", imagemRedimensionada[1760:1930, 350:680])
        cv2.imwrite("Quadrantes/10x2.jpg", imagemRedimensionada[1760:1930, 680:1000])

        # Linha 12
        cv2.imwrite("Quadrantes/11x0.jpg", imagemRedimensionada[1930:2100, 0:350])
        cv2.imwrite("Quadrantes/11x1.jpg", imagemRedimensionada[1930:2100, 350:680])
        cv2.imwrite("Quadrantes/11x2.jpg", imagemRedimensionada[1930:2100, 680:1000])

        # Linha 13
        cv2.imwrite("Quadrantes/12x0.jpg", imagemRedimensionada[2100:2270, 0:350])
        cv2.imwrite("Quadrantes/12x1.jpg", imagemRedimensionada[2100:2270, 350:680])
        cv2.imwrite("Quadrantes/12x2.jpg", imagemRedimensionada[2100:2270, 680:1000])

        # Linha 14
        cv2.imwrite("Quadrantes/13x0.jpg", imagemRedimensionada[2270:2450, 0:350])
        cv2.imwrite("Quadrantes/13x1.jpg", imagemRedimensionada[2270:2450, 350:680])
        cv2.imwrite("Quadrantes/13x2.jpg", imagemRedimensionada[2270:2450, 680:1000])

        # Linha 15
        cv2.imwrite("Quadrantes/14x0.jpg", imagemRedimensionada[2450:2630, 0:350])
        cv2.imwrite("Quadrantes/14x1.jpg", imagemRedimensionada[2450:2630, 350:680])
        cv2.imwrite("Quadrantes/14x2.jpg", imagemRedimensionada[2450:2630, 680:1000])

        # Linha 16
        cv2.imwrite("Quadrantes/15x0.jpg", imagemRedimensionada[2630:2800, 0:350])
        cv2.imwrite("Quadrantes/15x1.jpg", imagemRedimensionada[2630:2800, 350:680])
        cv2.imwrite("Quadrantes/15x2.jpg", imagemRedimensionada[2630:2800, 680:1000])

    # Criando o método que reconhece os formatos:
    def classificarFormato(self, quadrante):
        # Redimensionando a imagem e convertendo o seu formato.
        quadrante = cv2.resize(quadrante, (96, 96))
        quadrante = quadrante.astype("float") / 255.0
        quadrante = img_to_array(quadrante)
        quadrante = np.expand_dims(quadrante, axis=0)

        # Carregando os arquivos da rede neural convolucional.
        model = load_model("CNN/Arquivos/Formatos/FormatosModel.model")
        lb = pickle.loads(open("CNN/Arquivos/Formatos/FormatosPickle.pickle", "rb").read())

        # Realizando a predição da imagem e anexando-a na respectiva classe.
        probabilidade = model.predict(quadrante)[0]
        index = np.argmax(probabilidade)
        classificacao = lb.classes_[index]

        return classificacao

    # Criando o método que reconhece as setas:
    def classificarSeta(self, quadrante):
        # Redimensionando a imagem e convertendo o seu formato.
        quadrante = cv2.resize(quadrante, (96, 96))
        quadrante = quadrante.astype("float") / 255.0
        quadrante = img_to_array(quadrante)
        quadrante = np.expand_dims(quadrante, axis=0)

        # Carregando os arquivos da rede neural convolucional.
        model = load_model("CNN/Arquivos/Setas/SetasModel.model")
        lb = pickle.loads(open("CNN/Arquivos/Setas/SetasPickle.picklE", "rb").read())

        # Realizando a predição da imagem e anexando-a na respectiva classe.
        probabilidade = model.predict(quadrante)[0]
        index = np.argmax(probabilidade)
        classificacao = lb.classes_[index]

        return classificacao

    # Utilizando a API para reconhecer o texto dentro do quadrante.
    def extrairTexto(self, imagem, linguagem):
        reconhecerFluxograma = ReconhecerFluxograma()

        # Salvando a dimensão da imagem original.
        altura, largura, _ = imagem.shape

        # Carregando a URL da API para poder acessá-la.
        urlAPI = "https://api.ocr.space/parse/image"

        # Comprimindo o tamanho da imagem nos padrões aceitos da API.
        _, imagemCompactada = cv2.imencode(".jpg", imagem, [1, 90])
        arquivoCompactado = io.BytesIO(imagemCompactada)

        # Obtendo os resultados obtidos ao enviar o arquivo, a linguagem e a chave da API em formato .JSON"
        resultado = requests.post(urlAPI,
                                  files={"screenshot.jpg": arquivoCompactado},
                                  data={"apikey": "8f06577e5488957",
                                        "language": "por"})
        resultado = resultado.content.decode()
        resultado = json.loads(resultado)

        # Convertendo o resultado em uma STRING.
        resultadoAnalisado = resultado.get("ParsedResults")[0]
        textoExtraido = resultadoAnalisado.get("ParsedText")

        # Alterando algumas expressões para atender a linguagem de programação.
        textoExtraido = textoExtraido.strip()
        textoExtraido = textoExtraido.upper()
        textoExtraido = reconhecerFluxograma.editarTexto(textoExtraido, linguagem)

        return textoExtraido

    # Obtendo as áreas de texto para melhorar os resultados do OCR.
    def detectarTexto(self, quadrante, linguagem):
        imagemFinal = quadrante

        # Aplicando pré-processamentos para melhorar o texto: escala de cinza, binarização e dilatação.
        imagemCinza = cv2.cvtColor(quadrante, cv2.COLOR_BGR2GRAY)
        ret, mascara = cv2.threshold(imagemCinza, 180, 255, cv2.THRESH_BINARY)
        imagemFinal2 = cv2.bitwise_and(imagemCinza, imagemCinza, mask=mascara)
        ret, novaImagem = cv2.threshold(imagemFinal2, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                             3))
        imagemDilatada = cv2.dilate(novaImagem, kernel, iterations=1)

        # Obtendo os contornos da imagem.
        contornos, hierarquia = cv2.findContours(imagemDilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Criação das variáveis necessárias.
        reconhecerFluxograma = ReconhecerFluxograma()
        texto = ''

        # Percorrendo os contornos, desenhando retângulos ao redor daqueles de maior destaque.
        for contorno in contornos:
            [x, y, l, a] = cv2.boundingRect(contorno)

            # Condição para desenhar os retângulos somente nas coordenadas com um tamanho reduzido.
            if l < 35 and a < 35:
                continue

            # Recortando a imagem das coordenadas do retângulo, aplicando mais pré-processmanto: ampliação da imagem.
            imagemRecortada = imagemFinal[y:y + a, x: x + l]
            imagemRecortada = cv2.resize(imagemRecortada, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Concatenando o texto obtido em cada um dos recortes.
            texto += reconhecerFluxograma.extrairTexto(imagemRecortada)

        # Eliminando espaços em branco do texto obtido.
        textoFinal = texto.strip()

        # Alterando algumas expressões para atender a linguagem de programação.
        textoFinal = textoFinal.upper()
        textoFinal = reconhecerFluxograma.editarTexto(textoFinal, linguagem)

        return textoFinal

    # Instanciar as matrizes necessárias.
    def instanciarMatrizes(self):
        # Matrizes.
        global quadranteLinOrigem
        global quadranteColOrigem
        global quadranteLinDestino
        global quadranteColDestino
        global quadranteLinDestino2
        global quadranteColDestino2

        # Instanciando as matrizes com valores nulos.
        for i in range(LINMAX+1):
            linha1 = []
            linha2 = []
            linha3 = []
            linha4 = []
            linha5 = []
            linha6 = []

            for j in range(COLMAX+1):
                linha1.append(0)
                linha2.append(0)
                linha3.append(0)
                linha4.append(0)
                linha5.append(0)
                linha6.append(0)

            quadranteLinOrigem.append(linha1)
            quadranteColOrigem.append(linha2)
            quadranteLinDestino.append(linha3)
            quadranteColDestino.append(linha4)
            quadranteLinDestino2.append(linha5)
            quadranteColDestino2.append(linha6)

    # Método para a geração do código na linguagem Portugol.
    def gerarCodigo(self, linguagem):
        # Global.
        global LINMAX
        global COLMAX
        global inicioLin
        global inicioCol
        global fimLin
        global fimCol
        global fimCondicao
        global lin
        global col
        global variavelTipo

        # Matrizes.
        global quadranteLabel
        global quadranteTexto
        global quadranteLinOrigem
        global quadranteColOrigem
        global quadranteLinDestino
        global quadranteColDestino
        global quadranteLinDestino2
        global quadranteColDestino2

        # Varíaveis.
        reconhecerFluxograma = ReconhecerFluxograma()
        i = lin
        j = col
        i_prox = quadranteLinDestino[i][j]
        j_prox = quadranteColDestino[i][j]

        # Ajustando as conversões de tipagem para o Python.
        if variavelTipo == "inteiro":
            variavelTipo = "int"
        elif variavelTipo == "caractere":
            variavelTipo = "string"

        # Escrevendo as linhas de código a partir da classificação do quadrante.
        if quadranteLabel[i][j] == "saida":
            if linguagem == 1:
                reconhecerFluxograma.escreverLinha('escreva({})'.format(quadranteTexto[i][j]))
            elif linguagem == 2:
                reconhecerFluxograma.escreverLinha('print({})'.format(quadranteTexto[i][j]))

        elif quadranteLabel[i][j] == "entrada":
            if linguagem == 1:
                reconhecerFluxograma.escreverLinha('leia({})'.format(quadranteTexto[i][j]))
            elif linguagem == 2:
                if variavelTipo != "string":
                    reconhecerFluxograma.escreverLinha('{} = {}(input())'.format(quadranteTexto[i][j], variavelTipo))
                else:
                    reconhecerFluxograma.escreverLinha('{} = input()'.format(quadranteTexto[i][j]))

        elif quadranteLabel[i][j] == "processamento":
            reconhecerFluxograma.escreverLinha("{}".format(quadranteTexto[i][j]))

        elif quadranteLabel[i][j] == "terminal":
            if linguagem == 1:
                if quadranteTexto[i][j] == "FIM" or quadranteTexto[i][j] == "fim":
                    reconhecerFluxograma.escreverLinha("fim")
                else:
                    reconhecerFluxograma.escreverLinha("inicio")

        # Caso a estrutura seja uma decisão "if...then...", as linhas a serem escritas entrarão dentro desse laço.
        elif quadranteLabel[i][j] == "decisaoDois":
            fimCondicao = False
            if linguagem == 1:
                reconhecerFluxograma.escreverLinha("se ({}) entao".format(quadranteTexto[i][j]))
            elif linguagem == 2:
                reconhecerFluxograma.escreverLinha('if {}:'.format(quadranteTexto[i][j]))

            # Atribuindo as coordenadas do próximo quadrante a ser analisado.
            lin = quadranteLinDestino[i][j]
            col = quadranteColDestino[i][j]

            # Executando a geração do código do laço "if...then..." a partir da recursividade.
            while True:
                reconhecerFluxograma.gerarCodigo(linguagem)
                if fimCondicao:
                    if linguagem == 1:
                        reconhecerFluxograma.escreverLinha("fimse")
                    break
            return

        # Caso a estrutura seja uma decisão "if...then...else...", as linhas a serem escritas entrarão dentro desses
        # dois laços devidamente separadas.
        elif quadranteLabel[i][j] == "decisao":
            fimCondicao = False
            if linguagem == 1:
                reconhecerFluxograma.escreverLinha("se ({}) entao".format(quadranteTexto[i][j]))
            elif linguagem == 2:
                reconhecerFluxograma.escreverLinha('if {}:'.format(quadranteTexto[i][j]))

            # Atribuindo as coordenadas do próximo quadrante a ser analisado.
            lin = quadranteLinDestino[i][j]
            col = quadranteColDestino[i][j]

            # Executando a geração do código do laço "if...then..." a partir da recursividade.
            while True:
                reconhecerFluxograma.gerarCodigo(linguagem)
                if fimCondicao:
                    break

            if linguagem == 1:
                reconhecerFluxograma.escreverLinha("senao")
            elif linguagem == 2:
                reconhecerFluxograma.escreverLinha("else:")
            fimCondicao = False

            # Atribuindo as coordenadas.
            lin = quadranteLinDestino2[i][j]
            col = quadranteColDestino2[i][j]

            # Executando a geração do código do laço "else..." a partir da recursividade.
            while True:
                reconhecerFluxograma.gerarCodigo(linguagem)
                if fimCondicao:
                    if linguagem == 1:
                        reconhecerFluxograma.escreverLinha("fimse")
                    break
            return

        # Caso haja duas setas encontrando-se ortogonalmente, entende-se como a finalização do laço em análise.
        if ((quadranteLabel[i][j] == "direita" or quadranteLabel[i][j] == "esquerda") and (quadranteLabel[i_prox][
                                                                                               j_prox] == "cima" or
                                                                                           quadranteLabel[i_prox][
                                                                                               j_prox] == "baixo")) or (
                (quadranteLabel[i][j] == "cima" or
                 quadranteLabel[i][j] == "baixo") and (quadranteLabel[i_prox][j_prox] == "direita" or
                                                       quadranteLabel[i_prox][j_prox] == "esquerda")):
            fimCondicao = True
            # Atribuindo as coordenadas do próximo quadrante a ser analisado.
            lin = quadranteLinDestino[i_prox][j_prox]
            col = quadranteColDestino[i_prox][j_prox]

        else:
            # Atribuindo as coordenadas do próximo quadrante a ser analisado.
            lin = quadranteLinDestino[i][j]
            col = quadranteColDestino[i][j]

    # Escrevendo as linhas de código em um arquivo de texto.
    def escreverLinha(self, texto):
        arquivoCodigo = open('Códigos/Código.txt', 'r+')
        conteudoCodigo = arquivoCodigo.readlines()
        conteudoCodigo.append(texto)
        arquivoCodigo = open('Códigos/Código.txt', 'w+')
        arquivoCodigo.writelines(conteudoCodigo)
        arquivoCodigo.write("\n")
        print(texto)
        arquivoCodigo.close()

    # Mapeando o fluxo do fluxograma em matrizes auxiliares.
    def definirFluxo(self):
        # Global.
        global LINMAX
        global COLMAX
        global inicioLin
        global inicioCol
        global fimLin
        global fimCol
        global fimCondicao
        global lin
        global col

        # Matrizes.
        global quadranteLabel
        global quadranteTexto
        global quadranteLinOrigem
        global quadranteColOrigem
        global quadranteLinDestino
        global quadranteColDestino
        global quadranteLinDestino2
        global quadranteColDestino2

        # Condição para atualização do número de linha e de coluna a ser analisado.
        if col > COLMAX:
            col = 0
            lin += 1
            if lin > LINMAX:
                return

        # Variáveis.
        i = lin
        j = col

        print("[INFO] Quadrante [{}]x[{}] classificado em: {}.".format(i, j, quadranteLabel[i][j]))

        # Mapeando o fluxograma a partir da posição em análise e das classificações dos quadrantes ao redor desta.
        if quadranteLabel[i][j] != "vazio":
            if quadranteLabel[i][j] == "terminal" and quadranteTexto[i][j] != "FIM":
                inicioLin = i
                inicioCol = j

            elif quadranteLabel[i][j] == "terminal" and quadranteTexto[i][j] == "FIM":
                fimLin = i
                fimCol = j

            elif quadranteLabel[i][j] == "cima":
                if lin < LINMAX:
                    quadranteLinOrigem[i][j] = i + 1
                    quadranteColOrigem[i][j] = j

                if lin > 0:
                    quadranteLinDestino[i][j] = i - 1
                    quadranteColDestino[i][j] = j

            elif quadranteLabel[i][j] == "baixo":
                if lin > 0:
                    if quadranteLabel[i][j - 1] == "decisao" or quadranteLabel[i][j - 1] == "decisaoDois":
                        quadranteLinOrigem[i][j] = i
                        quadranteColOrigem[i][j] = j - 1

                    elif quadranteLabel[i][j - 1] == "direita":
                        quadranteLinOrigem[i][j] = i
                        quadranteColOrigem[i][j] = j - 1

                    elif quadranteLabel[i][j + 1] == "esquerda":
                        quadranteLinOrigem[i][j] = i
                        quadranteColOrigem[i][j] = j + 1

                    else:
                        quadranteLinOrigem[i][j] = i - 1
                        quadranteColOrigem[i][j] = j

                if lin < LINMAX:
                    quadranteLinDestino[i][j] = i + 1
                    quadranteColDestino[i][j] = j

            elif quadranteLabel[i][j] == "direita":
                if col > 0:
                    quadranteLinOrigem[i][j] = i
                    quadranteColOrigem[i][j] = j - 1

                if col < COLMAX:
                    quadranteLinDestino[i][j] = i
                    quadranteColDestino[i][j] = j + 1

            elif quadranteLabel[i][j] == "esquerda":
                if col < COLMAX:
                    quadranteLinOrigem[i][j] = i
                    quadranteColOrigem[i][j] = j + 1

                if col > 0:
                    quadranteLinDestino[i][j] = i
                    quadranteColDestino[i][j] = j - 1

            elif quadranteLabel[i][j] == "decisao" or quadranteLabel[i][j] == "decisaoDois":
                if lin > 0 and quadranteLabel[i - 1][j] == "baixo":
                    quadranteLinOrigem[i][j] = i - 1
                    quadranteColOrigem[i][j] = j

                if lin < LINMAX and quadranteLabel[i + 1][j] == "cima":
                    quadranteLinOrigem[i][j] = i + 1
                    quadranteColOrigem[i][j] = j

                if quadranteLabel[i][j] == "decisao" and col > 0 and quadranteLabel[i][j - 1] == "direita":
                    quadranteLinOrigem[i][j] = i
                    quadranteColOrigem[i][j] = j - 1

                if quadranteLabel[i][j] == "decisao" and col < COLMAX and quadranteLabel[i][j + 1] == "esquerda":
                    quadranteLinOrigem[i][j] = i
                    quadranteColOrigem[i][j] = j + 1

                if lin > 0 and quadranteLabel[i - 1][j] == "cima":
                    quadranteLinDestino[i][j] = i - 1
                    quadranteColDestino[i][j] = j

                if lin < LINMAX and quadranteLabel[i + 1][j] == "baixo":
                    quadranteLinDestino[i][j] = i + 1
                    quadranteColDestino[i][j] = j

                if col < COLMAX and quadranteLabel[i][j + 1] == "direita" or quadranteLabel[i][j + 1] == "baixo":
                    quadranteLinDestino2[i][j] = i
                    quadranteColDestino2[i][j] = j + 1
                    if quadranteLabel[i][j + 1] == "baixo":
                        quadranteLinOrigem[i][j + 1] = i
                        quadranteColOrigem[i][j + 1] = j

                if col > 0 and quadranteLabel[i][j - 1] == "esquerda" or quadranteLabel[i][j - 1] == "baixo":
                    quadranteLinDestino[i][j] = i
                    quadranteColDestino[i][j] = j - 1
                    if quadranteLabel[i][j - 1] == "baixo":
                        quadranteLinOrigem[i][j - 1] = i
                        quadranteColOrigem[i][j - 1] = j

            elif quadranteLabel[i][j] == "saida" or quadranteLabel[i][j] == "entrada" or quadranteLabel[i][
                j] == "processamento":
                if lin > 0:
                    if quadranteLabel[i - 1][j] == "baixo":
                        quadranteLinOrigem[i][j] = i - 1
                        quadranteColOrigem[i][j] = j

                    else:
                        if quadranteLabel[i - 1][j] == "cima":
                            quadranteLinDestino[i][j] = i - 1
                            quadranteColDestino[i][j] = j

                if lin < LINMAX:
                    if quadranteLabel[i + 1][j] == "baixo":
                        quadranteLinDestino[i][j] = i + 1
                        quadranteColDestino[i][j] = j

                    elif quadranteLabel[i + 1][j] == "esquerda":
                        quadranteLinDestino[i][j] = i + 1
                        quadranteColDestino[i][j] = j

                    elif quadranteLabel[i + 1][j] == "direita":
                        quadranteLinDestino[i][j] = i + 1
                        quadranteColDestino[i][j] = j

                    else:
                        if quadranteLabel[i + 1][j] == "cima":
                            quadranteLinOrigem[i][j] = i + 1
                            quadranteColOrigem[i][j] = j

                if col > 0:
                    if quadranteLabel[i][j - 1] == "direita":
                        quadranteLinOrigem[i][j] = i
                        quadranteColOrigem[i][j] = j - 1

                    else:
                        if quadranteLabel[i][j - 1] == "esquerda":
                            quadranteLinDestino[i][j] = i
                            quadranteColDestino[i][j] = j - 1

                if col < COLMAX and col + 1 < 3:
                    if quadranteLabel[i][j + 1] == "direita":
                        quadranteLinDestino[i][j] = i
                        quadranteColDestino[i][j] = j - 1

                    else:
                        if quadranteLabel[i][j + 1] == "esquerda":
                            quadranteLinOrigem[i][j] = i
                            quadranteColOrigem[i][j] = j - 1

            if quadranteLinOrigem[i][j] == inicioLin and quadranteColOrigem[i][j] == inicioCol:
                quadranteLinDestino[inicioLin][inicioCol] = i
                quadranteColDestino[inicioLin][inicioCol] = j

            if quadranteLinDestino[i][j] == fimLin and quadranteColDestino[i][j] == fimCol:
                quadranteLinOrigem[fimLin][fimCol] = i
                quadranteColOrigem[fimLin][fimCol] = j

        # Condição para atualização do número de linha e de coluna a ser analisado.
        if col < COLMAX+1:
            col += 1
        else:
            lin += 1
            col = 0

    # Executando a tradução do fluxograma a partir das coordenadas obtidas no mapeamento, gerando as linhas de código
    # respectivas.
    def executarTraducao(self, quantidadeImagens, linguagem):
        # Global.
        global LINMAX
        global COLMAX
        global inicioLin
        global inicioCol
        global fimLin
        global fimCol
        global lin
        global col

        # Matrizes.
        global quadranteLabel
        global quadranteTexto

        # Variáveis.
        reconhecerFluxograma = ReconhecerFluxograma()

        if quantidadeImagens == 2:
            LINMAX = 15
        else:
            LINMAX = 7

        reconhecerFluxograma.instanciarMatrizes()

        for i in range(LINMAX+1):
            # Arrays para instanciar as matrizes de classificação.
            classificacaoQuadrante = []
            classificacaoQuadranteAux = []
            textoQuadrante = []

            for j in range(COLMAX+1):
                nomeImagem = "Quadrantes/{}x{}.jpg".format(i, j)

                # Classificando um quadrante como vazio ou não de acordo com a ausência de texto.
                textoOCR = reconhecerFluxograma.extrairTexto(cv2.imread(nomeImagem), linguagem)

                # Acrescentando uma função de tempo para tentar melhorar o rendimento da API.
                time.sleep(5.0)

                if textoOCR == "":
                    classificacaoQuadrante.append(reconhecerFluxograma.classificarSeta(cv2.imread(nomeImagem)))
                else:
                    classificacaoQuadrante.append(reconhecerFluxograma.classificarFormato(cv2.imread(nomeImagem)))

                textoQuadrante.append(textoOCR)

            quadranteTexto.append(textoQuadrante)
            quadranteLabel.append(classificacaoQuadrante)

        # Mapeando o fluxograma a partir das classificaçoes inseridas nas matrizes.
        print("\n[INFO] Mapeando o fluxograma:\n")
        while True:
            reconhecerFluxograma.definirFluxo()
            if not lin <= LINMAX:
                break

        print("\n[INFO] Matrizes instanciadas:")
        print("[ >> ] quadranteLinDestino:", quadranteLinDestino, "\n[ >> ] quadranteColDestino:", quadranteColDestino,
              "\n[ >> ] quadranteLinDestino2:", quadranteLinDestino2, "\n[ >> ] quadranteColDestino2:",
              quadranteColDestino2, "\n[ >> ] quadranteLinOrigem:", quadranteLinOrigem, "\n[ >> ] quadranteColOrigem:",
              quadranteColOrigem, "\n[ >> ] quadranteLabel:", quadranteLabel, "\n[ >> ] quadranteTexto:", quadranteTexto, "\n[ >> ] inicioLin:")

        lin = 0
        col = 1

        print("\n[INFO] Código sendo gerado...\n")

        # Gerando as linhas de código dos quadrantes mapeados.
        while True:
            reconhecerFluxograma.gerarCodigo(linguagem)
            if lin == fimLin and col == fimCol:
                break

        print("\n[INFO] Fim da geração.")

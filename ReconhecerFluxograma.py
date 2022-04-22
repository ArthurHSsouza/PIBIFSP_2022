# Realizando as importações necessárias:
from ClasseReconhecerFluxograma import ReconhecerFluxograma
import cv2

input("[ >> ] Bem vindo ao programa de reconhecimento de fluxogramas. Pressione qualquer tecla para dar início ao "
      "processo.")

# Carregando a classe com os métodos.
reconhecerFluxograma = ReconhecerFluxograma()

# Pegando a quantidade de imagens para concatená-las ou não.
while True:
    quantidadeImagens = int(input("\n[ >> ] Digite a quantidade imagens que compõem o fluxograma:"))
    if 0 < quantidadeImagens <= 2:
        break

# Lendo o caminho das imagens.
if quantidadeImagens == 1:
    print("\n[ >> ] Coloque o caminho da imagem para análise:")
    # Fluxogramas/Teste(1.1).png
    imagem = cv2.imread(input())
    imagem = cv2.resize(imagem, (1000, 1400), interpolation=cv2.INTER_AREA)

else:
    print("\n[ >> ] Coloque os caminhos da imagem para análise:")
    imagem1 = cv2.imread(input("[ >> ] Primeira imagem:"))
    imagem2 = cv2.imread(input("[ >> ] Segunda imagem:"))
    imagem = cv2.vconcat([imagem1, imagem2])
    imagem = cv2.resize(imagem, (1000, 2800), interpolation=cv2.INTER_AREA)

print("\n[INFO] Imagem selecionada. Iniciando extração dos quadrantes.")
# Recortando os quadrantes do painel, salvando-os no disco para o reconhecer o formato e o texto neles presente.
if quantidadeImagens == 1:
    reconhecerFluxograma.recortarQuadrantes1(imagem)
else:
    reconhecerFluxograma.recortarQuadrantes2(imagem)

# Criando arquivo de texto em que o código será escrito.
print("[INFO] Criando arquivo de texto que conterá o código gerado. Você poderá acessá-lo no fim dessa execução.")
try:
    codigo = open('Códigos/Código.txt', 'r+')
except FileNotFoundError:
    codigo = open('Códigos/Código.txt', 'w+')
    codigo.close()

# Consulta para saber a opção de linguagem do código.
while True:
    linguagem = int(input("\n[ >> ] Qual a linguagem desejada para a geração do código? Escolha '1' para Portugol ou "
                          "'2' "
                          "para Python:"))
    if linguagem == 1 or linguagem == 2:
        break

print("\n[INFO] As mensagens a seguir contêm os relatórios de execução para fins de testes. Por favor, ignore-as.")
input("[ >> ] Pressione qualquer tecla para continuar.\n")

# Executando os métodos para geração do código.
reconhecerFluxograma.executarTraducao(quantidadeImagens, linguagem)



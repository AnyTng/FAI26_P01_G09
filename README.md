# FAI26_P01_G09

Projeto 01 de Fundamentos de Inteligência Artificial do grupo G09, de Gonçalo Tierri (23020) e Ricardo Oliveira (20432).

## Habitação — classes de preço e árvore de decisão

Este trabalho usa o [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot).
O preço observado de venda (`Price`) é agrupado em **barato**, **normal** e **caro** com
`sklearn.cluster.KMeans(n_clusters=3)`. As classes são acrescentadas à tabela e previstas
por uma árvore de classificação **implementada de raiz**.

«Barato» significa pertencer ao grupo de preços mais baixos desta base; não significa
estar subavaliado para a zona. Os dados são de Melbourne, não de Portugal.

### Ficheiros

- `01_eda.ipynb`: qualidade, distribuições e fundamentação da variável-alvo.
- `02_classify.ipynb`: criação das classes, tabela classificada, ajuste e avaliação da árvore.
- `03_clusters.ipynb`: análise separada de perfis com K-Means de raiz, sem usar o preço.
- `libs/data.py`: carregamento, classes de preço, separação por imóvel e preparação.
- `libs/decisiontrees.py`: árvore multiclasse por Gini.
- `libs/kmeans.py` e `libs/metrics.py`: K-Means e métricas de raiz.
- `reports/housing_*`: resultados reais, classes atribuídas e configurações avaliadas.
- `notebooks/`, `src/`, `data_tiny/` e `data_small/`: planeamento de rotas do Projeto 01.

### Método e proteção da avaliação

1. Separar aproximadamente 60/20/20 por `Suburb + Address`: vendas da mesma morada
   ficam no mesmo conjunto. Foram encontrados 114 registos adicionais de moradas repetidas.
2. Aprender três grupos usando **apenas Price do treino**. Ordenar os centroides para
   atribuir os nomes; IDs arbitrários de clusters não têm significado económico.
3. Aplicar essa definição à validação e ao teste, criando `PriceClass` e `ClassePreco`.
4. Aprender medianas e categorias apenas no treino. Excluir `Price`, classes,
   morada, identificadores, vendedor, data e método de venda das entradas da árvore.
5. Escolher entre 12 configurações de `max_depth`, `min_samples_split` e
   `min_samples_leaf` pela F1 macro da validação. Não escolher ajustes pelo teste.
6. Avaliar o modelo escolhido no teste e comparar com a classe maioritária do treino.

Por orientação recebida, scikit-learn é usado para criar a variável-alvo.
A árvore e o K-Means exploratório do notebook 03 são de raiz.
Contribuições individuais: preencher com o trabalho efetivamente realizado.

### Primeira execução real

Foram utilizadas as 13 580 linhas do CSV; não havia cópias exatas nem preços inválidos.
Separação: 8 149 treino, 2 716 validação e 2 715 teste.

| Classe | Centro do grupo no treino | Fronteiras aproximadas no preço original | Casas na base completa |
| --- | ---: | --- | ---: |
| barato | 701 455,90 | até 1 083 721,94 | 8 438 |
| normal | 1 465 987,98 | de 1 083 721,94 a 2 162 281,58 | 4 322 |
| caro | 2 858 575,17 | acima de 2 162 281,58 | 820 |

Os valores estão nas unidades monetárias originais; não foram convertidos para euros.
As fronteiras são os pontos médios dos centroides; em empate exato, a atribuição segue o modelo.

Configuração escolhida: profundidade 7, mínimo de 20 exemplos para dividir e 10 por folha.

| Modelo | Accuracy no teste | F1 macro no teste |
| --- | ---: | ---: |
| Classe maioritária | 61,58% | 0,2541 |
| Árvore de raiz | 80,41% | 0,6542 |

**Limitação importante:** apenas 38 de 169 casas caras foram identificadas como caras
(recall de 22,49%); 129 foram classificadas como normais. A accuracy global não resolve
esta dificuldade. Não se ajustou uma nova configuração a partir destes erros do teste.

No clustering exploratório, K=3 teve a maior silhouette entre K=2 e K=6:
0,3254 numa amostra fixa de 500 pontos. Os grupos quase reproduzem o tipo do imóvel,
mostrando forte influência da codificação de `Type`; não são as três classes de preço.

### Executar e obter a tabela

Python 3.11 ou superior, num ambiente virtual:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Abrir os notebooks na raiz do repositório. O CSV público é descarregado apenas se
`data/melbourne/melb_data.csv` não existir; também se pode definir `MELBOURNE_DATA`
com a pasta de origem.

O notebook 02 grava a tabela completa com as novas colunas em
`data/melbourne/melb_classificado.csv`. Os CSV completos ficam fora do Git.
A versão resumida publicada é `reports/housing_price_classes.csv`, identificada por
`SourceRow`, o índice da linha no CSV original (começando em zero).

`reports/housing_results.json` regista o hash da fonte, versões do ambiente,
fronteiras, variáveis, parâmetros e métricas. Os notebooks incluem outputs reais.

### Validação e questões em aberto

```bash
python -m unittest discover -s tests -v
```

Os testes verificam as três classes, a ausência de fuga de informação na sua criação,
a separação de moradas, a preparação, as métricas e os algoritmos.
Os três notebooks foram executados integralmente com o CSV original.

Rever os valores atípicos e a sua interpretação: existem áreas zero, grandes extremos
e `YearBuilt=1196`. Não se corrigiram valores por suposição nem se apagaram extremos
automaticamente. Área zero pode ter significados diferentes conforme o tipo de imóvel.

Antes da entrega: rever com o docente o alcance do uso de bibliotecas e das regras de
associação, preencher contribuições e preparar ZIP/apresentação.
O guião indica um repositório privado com nomenclatura `FAI26_P02_G##`.

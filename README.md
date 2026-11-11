# FAI26_P01_G09

Projeto 01 de Fundamentos de Inteligência Artificial do grupo G09, de Gonçalo Tierri (23020) e Ricardo Oliveira (20432).

## Home Credit — trabalho do Projeto 02

Os notebooks de planeamento de rotas do Projeto 01 mantêm-se em `notebooks/`.
Os notebooks de Home Credit descritos abaixo estão na raiz deste repositório.

Projeto 02 de Fundamentos de Inteligência Artificial, Grupo 09, MIAA, UPCA.

| Nome | Número |
| --- | --- |
| Gonçalo Tierri | 23020 |
| Ricardo Oliveira | 20432 |

Contribuições individuais: preencher após a divisão e realização efetiva das tarefas.

### Estado e objetivos

Implementação inicial para duas perguntas:
- Classificação: prever dificuldades de pagamento segundo a definição de `TARGET`.
- Clustering: descobrir perfis financeiros de pedidos sem usar `TARGET`.

Ainda não há resultados reais calculados ou conclusões validadas.
O dataset não é especificamente português e não garante que os empréstimos sejam para habitação.

### Requisitos e decisões

O guião pede EDA, árvore de decisão de raiz com pelo menos três hiperparâmetros,
K-Means de raiz, avaliação, interpretação e defesa do trabalho.

As implementações estão em `libs/decisiontrees.py` e `libs/kmeans.py`.
Usamos NumPy/pandas para operações e preparação e Matplotlib para gráficos;
não usamos modelos nem métricas prontas de bibliotecas de ML.

A seleção de sete variáveis numéricas, a amostra máxima inicial de 20 000 pedidos,
a grelha de hiperparâmetros e os K de 2 a 8 são propostas de desenvolvimento.
Serão revistos após a EDA e medição dos tempos, sem usar o desempenho no teste para escolher ajustes.

Confirmar com o docente:
1. Autorização para NumPy, pandas e Matplotlib como auxiliares.
2. Adequação temática deste dataset de crédito.
3. Obrigatoriedade de regras de associação, mencionadas apenas na introdução.
4. Se os três notebooks devem ser reunidos num único notebook na entrega.

### Executar

Python 3.11 ou superior, num ambiente virtual:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Abrir os notebooks a partir da raiz do repositório:
1. `01_eda.ipynb`: dicionário, qualidade, distribuição das classes e variáveis.
2. `02_classify.ipynb`: separação estratificada 60/20/20, preparação aprendida no treino,
   pesquisa dos três hiperparâmetros e limiar na validação, comparação final com baseline no teste.
3. `03_clusters.ipynb`: preparação própria, comparação de K, silhouette amostral,
   diagnóstico entre sementes e descrição dos perfis.

Os três notebooks podem ser executados de forma independente.
Não instalam dependências automaticamente nem descarregam dados por defeito.

### Dados

Fonte original: [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data).

São necessários apenas:
- `application_train.csv`
- `HomeCredit_columns_description.csv`

Colocar ambos em `data/home_credit/`, ou definir a variável de ambiente `HOME_CREDIT_DATA`
com a pasta que os contém. Os CSV permanecem fora do Git.

Também é possível usar `DOWNLOAD = True` na primeira célula de código.
O carregador usa [kagglehub.competition_download](https://github.com/Kaggle/kagglehub#download-competition)
para obter apenas os dois ficheiros.
A conta Kaggle deve estar autenticada e o utilizador deve aceitar previamente as regras
da competição. Credenciais nunca devem ser guardadas no código ou no repositório.

`application_test.csv` não tem os rótulos públicos necessários à avaliação local.
A nossa separação de teste é criada a partir de `application_train.csv`.

Não há fallback para dados artificiais quando os dados reais estão ausentes.
Verificar as condições da competição antes de redistribuir os CSV no ZIP da entrega.

### Validação e limitações

```bash
python -m unittest discover -s tests -v
```

Os testes usam casos pequenos artificiais com resultados conhecidos, exclusivamente para
validar o código. Não são experiências de Home Credit nem evidência de qualidade preditiva.

A árvore usa divisões binárias por Gini e os hiperparâmetros `max_depth`,
`min_samples_split` e `min_samples_leaf`. Variáveis categóricas ficam fora desta primeira versão.
Precisão e F1 são zero quando não há previsões positivas.

O K-Means usa Lloyd, inicialização k-means++, várias execuções e tratamento de grupos vazios.
A silhouette usa uma subamostra fixa de até 500 pontos para limitar o custo quadrático.
A maior silhouette propõe um K, mas não substitui interpretação e análise de estabilidade.
`TARGET` e identificadores não entram nas distâncias.

Antes da entrega: executar com os CSV reais, interpretar resultados, registar versões das
dependências, preencher contribuições, rever o guião e verificar o ZIP.


# FAI26_P01_G09: agente de recolha e entrega de alimentos locais

Projeto 01 de Fundamentos de Inteligência Artificial (Mestrado em Inteligência Artificial Aplicada, 2026-27), grupo G09.

O agente planeia as rotas de uma frota que recolhe produtos em pequenos produtores do distrito de Braga, passa pelo centro de recolha e entrega as encomendas aos clientes, respeitando as janelas temporais, a capacidade dos veículos e a permanência máxima no armazém. Resolvemos o problema com pesquisa informada (Greedy, A\* e A\* ponderado, com quatro heurísticas), pesquisa local (Hill Climbing e Simulated Annealing) e um CSP (backtracking com MRV, forward checking, AC-3 e branch & bound).

O documento principal é [`notebooks/01_Project01_G09.ipynb`](notebooks/01_Project01_G09.ipynb).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/tiny`, `data/small` | CSV originais dos datasets, sem alterações |
| `src/model.py` | entidades, carregamento, simulador, custo, viabilidade, auditoria dos dados, verificador linha a linha e figuras |
| `src/informed.py` | Greedy, A\*, A\* ponderado, A\* por viagens e heurísticas h1–h4 |
| `src/local_search.py` | Hill Climbing e Simulated Annealing |
| `src/csp.py` | formulação CSP e algoritmos |
| `run_experiments.py` | corre todas as experiências e grava os resultados em `reports/` |
| `reports/` | resultados (CSV e JSON) que os notebooks leem |
| `notebooks/01_Project01_G09.ipynb` | notebook principal, do enunciado à conclusão |
| `notebooks/02_eda_tiny.ipynb` | análise exploratória do tiny, com uma comparação final com o small |
| `notebooks/03_informed.ipynb` | pesquisa informada |
| `notebooks/04_local_search.ipynb` | pesquisa local |
| `notebooks/05_csp.ipynb` | CSP |
| `tests/` | testes unitários |

Os notebooks 03 a 05 incluem a descrição do problema e a formulação da respetiva técnica, por isso podem ser lidos à parte.

## Como executar

É preciso Python 3.11 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Testes:

```bash
python -m unittest discover -s tests -t .
```

Todas as experiências (o A\* exato no tiny demora vários minutos por heurística):

```bash
python run_experiments.py
```

Uma versão rápida, com orçamentos pequenos:

```bash
python run_experiments.py --quick
```

Depois, abrir os notebooks com `jupyter lab` e fazer *Restart Kernel and Run All*. As pesquisas mais longas são lidas de `reports/`; com `RUN_HEAVY = True` voltam a correr dentro do notebook.

## Decisões de modelação

As linhas do mesmo produtor cujas janelas se intersetam formam um pedido de recolha, e cada encomenda forma um pedido de entrega. Uma rota é uma sequência de viagens que saem do centro e voltam lá; cada viagem só tem recolhas ou só tem entregas, e nenhuma passa a meia-noite. Nada é recolhido antes de `order_datetime`, as janelas limitam a hora a que o serviço começa e os veículos saem a tempo de chegar à primeira paragem quando a janela abre. Um veículo pode entregar o que outro recolheu, depois de descarregado no armazém, onde nada fica mais de 24 h. O objetivo é minimizar os km totais.

A discussão completa, com as limitações, está no notebook principal.

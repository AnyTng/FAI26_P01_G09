# FAI26_P01_G09 — Agente de recolha e entrega de alimentos locais

Projeto 01 de **Fundamentos de Inteligência Artificial** (Mestrado em Inteligência Artificial Aplicada, 2026-27), grupo **G09**.

Um agente planeia as rotas de uma frota que recolhe produtos em pequenos produtores do distrito de Braga, passa pelo centro de recolha (armazém) e entrega as encomendas aos clientes. O agente respeita as janelas temporais, a capacidade dos veículos e a permanência máxima no armazém. O problema é resolvido com:

- **pesquisa informada**: Greedy best-first, A\* e A\* ponderado, com quatro heurísticas admissíveis;
- **pesquisa local**: Hill Climbing (*steepest* e *first improvement*) e Simulated Annealing;
- **CSP**: backtracking com MRV, ordenação de valores para a restrição suave, forward checking, AC-3 e branch & bound.

O documento principal é [`notebooks/01_Project01_G09.ipynb`](notebooks/01_Project01_G09.ipynb).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/tiny`, `data/small` | CSV originais dos datasets (sem alterações) |
| `src/model.py` | Entidades (`Item`, `Request`, `Vehicle`, `Problem`, `State`), carregamento, simulador, custo e verificação de viabilidade. Inclui também a auditoria dos dados, o verificador independente linha a linha sobre os CSV, e as figuras (mapas, horários, curvas de custo) |
| `src/informed.py` | Greedy, A\*, A\* ponderado e heurísticas h1, h2, h3, h4 |
| `src/local_search.py` | Hill Climbing e Simulated Annealing |
| `src/csp.py` | Formulação CSP e algoritmos |
| `run_experiments.py` | Corre todas as experiências e grava os resultados em `reports/` |
| `reports/` | Resultados (CSV/JSON) lidos pelos notebooks |
| `notebooks/01_Project01_G09.ipynb` | Notebook principal: introdução, EDA, modelação, algoritmos, análise e conclusão |
| `notebooks/02_eda_tiny.ipynb` | Análise exploratória detalhada |
| `notebooks/03_informed.ipynb` | Pesquisa informada: heurísticas, nós expandidos, A\* ponderado, rotas do tiny |
| `notebooks/04_local_search.ipynb` | Pesquisa local: vizinhança, curvas de custo, sensibilidade do SA, solução inicial |
| `notebooks/05_csp.ipynb` | CSP: variáveis, domínios, restrições, configurações, backtracks, branch & bound |
| `tests/` | Testes unitários (fixture sintético + datasets) |

## Como executar

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para correr os testes:

```bash
python -m unittest discover -s tests -t .
```

Para reproduzir todas as experiências (o A\* exato no tiny demora vários minutos por heurística):

```bash
python run_experiments.py
```

Para uma verificação rápida, com orçamentos pequenos:

```bash
python run_experiments.py --quick
```

Depois, abrir os notebooks (`jupyter lab`) e fazer *Restart Kernel and Run All*. O notebook principal lê os resultados das pesquisas longas em `reports/`. Para voltar a correr essas pesquisas no próprio notebook, pôr `RUN_HEAVY = True`.

## Principais decisões de modelação

- As linhas do mesmo produtor cujas janelas se intersetam formam **um pedido de recolha**, e cada encomenda forma **um pedido de entrega**.
- Uma rota é uma sequência de **viagens** a partir do centro. Cada viagem só tem recolhas ou só tem entregas, e nenhuma atravessa a meia-noite.
- Não se recolhe nada antes de `order_datetime`. As janelas limitam o início do serviço, e os veículos saem a tempo de chegar quando a primeira janela abre (*just in time*).
- Um veículo pode entregar o que outro recolheu, depois de descarregado no armazém. A permanência máxima no armazém é de 24 h.
- O objetivo é minimizar os km totais.

Os detalhes e a discussão das limitações estão no notebook principal.

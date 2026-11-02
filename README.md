# FAI26_P01_G09: agente de recolha e entrega de alimentos locais

Projeto 01 de Fundamentos de Inteligência Artificial (Mestrado em Inteligência Artificial Aplicada, 2026-27), grupo G09.

O objetivo é planear as rotas de uma frota. A frota recolhe produtos em pequenos produtores do distrito de Braga, passa pelo centro de recolha e entrega as encomendas aos clientes. As rotas têm de respeitar as janelas temporais, a capacidade dos veículos e a permanência máxima no armazém.

Resolvemos o problema com três técnicas:

- pesquisa informada: Greedy, A\* e A\* ponderado, com quatro heurísticas
- pesquisa local: Hill Climbing e Simulated Annealing
- CSP: backtracking com MRV, forward checking, AC-3 e branch & bound

O documento principal é [`notebooks/01_Project01_G09.ipynb`](notebooks/01_Project01_G09.ipynb).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/tiny`, `data/small` | CSV originais dos datasets, sem alterações |
| `src/model.py` | entidades, carregamento, simulador, custo, viabilidade, auditoria dos dados, verificador linha a linha e figuras |
| `src/informed.py` | Greedy, A\*, A\* ponderado, A\* por viagens e heurísticas h1–h4 |
| `src/local_search.py` | Hill Climbing e Simulated Annealing |
| `src/csp.py` | formulação CSP e algoritmos |
| `notebooks/01_Project01_G09.ipynb` | notebook principal, do enunciado à conclusão |
| `notebooks/02_eda_tiny.ipynb` | análise exploratória do tiny, com uma comparação final com o small |
| `notebooks/03_informed.ipynb` | pesquisa informada |
| `notebooks/04_local_search.ipynb` | pesquisa local |
| `notebooks/05_csp.ipynb` | CSP |
| `tests/` | testes unitários |

Os notebooks 03 a 05 incluem a descrição do problema e a formulação da técnica. Podem ser lidos à parte.

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

Notebooks: abrir com `jupyter lab` e fazer *Restart Kernel and Run All*. Os notebooks 01 e 03 demoram cerca de 15 minutos, quase todos no A\* paragem a paragem.

## Decisões de modelação

- As linhas do mesmo produtor cujas janelas se intersetam formam um pedido de recolha. Cada encomenda é um pedido de entrega.
- Uma rota é uma sequência de viagens que saem do centro e voltam lá. Cada viagem só tem recolhas ou só tem entregas, e nenhuma passa a meia-noite.
- Nada é recolhido antes de `order_datetime`. As janelas limitam a hora a que o serviço começa.
- Os veículos saem a tempo de chegar à primeira paragem quando a janela abre.
- Um veículo pode entregar o que outro recolheu, depois de descarregado no armazém. Nada fica no armazém mais de 24 h.
- O objetivo é minimizar os km totais.

A discussão completa, com as limitações, está no notebook principal.

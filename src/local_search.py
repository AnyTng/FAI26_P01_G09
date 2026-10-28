"""Pesquisa local sobre soluções completas: Hill Climbing (steepest / first improvement) e
Simulated Annealing.

Vizinhança (todos os movimentos mantêm as viagens homogéneas):
  * relocate  - passa um pedido para outra posição, viagem ou veículo, ou para uma viagem nova;
  * swap      - troca dois pedidos do mesmo tipo;
  * 2-opt     - inverte um segmento dentro de uma viagem.
As soluções são avaliadas com ``Evaluation.penalized`` (km + penalizações). O Hill Climbing,
partindo de uma solução viável, só passa para vizinhos viáveis; o Simulated Annealing pode passar
por soluções inviáveis, mas devolve a melhor solução VIÁVEL que visitou.
"""
from __future__ import annotations

import copy
import math
import random
import time as _time
from dataclasses import dataclass, field

from .model import Problem, Solution, evaluate


def _clean(sol: Solution) -> Solution:
    return {v: [t for t in trips if t] for v, trips in sol.items()}


def _kind(p: Problem, trip) -> str:
    return p.requests[trip[0]].kind


def neighbours(p: Problem, sol: Solution):
    """Gera (descrição do movimento, solução vizinha) para a vizinhança inteira."""
    vids = sorted(sol)
    where = [(v, k, i, r) for v in vids for k, trip in enumerate(sol[v]) for i, r in enumerate(trip)]

    # relocate: mudar um pedido de sítio
    for v, k, i, r in where:
        kind = p.requests[r].kind
        base = copy.deepcopy(sol)
        del base[v][k][i]
        for v2 in vids:
            for k2, trip in enumerate(base[v2]):
                if trip and _kind(p, trip) != kind:
                    continue
                for j in range(len(trip) + 1):
                    if v2 == v and k2 == k and j == i:
                        continue
                    n = copy.deepcopy(base)
                    n[v2][k2].insert(j, r)
                    yield ("relocate", r, v2, k2, j), _clean(n)
            for k2 in range(len(base[v2]) + 1):                  # abrir uma viagem nova
                if v2 == v and len(sol[v][k]) == 1 and k2 in (k, k + 1):
                    continue                                     # a mesma solução
                n = copy.deepcopy(base)
                n[v2].insert(k2, [r])
                yield ("new-trip", r, v2, k2), _clean(n)

    # swap: trocar dois pedidos
    for a in range(len(where)):
        for b in range(a + 1, len(where)):
            va, ka, ia, ra = where[a]
            vb, kb, ib, rb = where[b]
            if p.requests[ra].kind != p.requests[rb].kind or (va, ka) == (vb, kb) and abs(ia - ib) == 0:
                continue
            n = copy.deepcopy(sol)
            n[va][ka][ia], n[vb][kb][ib] = rb, ra
            yield ("swap", ra, rb), n

    # 2-opt dentro de uma viagem
    for v in vids:
        for k, trip in enumerate(sol[v]):
            for i in range(len(trip) - 1):
                for j in range(i + 2, len(trip) + 1):
                    n = copy.deepcopy(sol)
                    n[v][k][i:j] = reversed(n[v][k][i:j])
                    yield ("2-opt", v, k, i, j), n


def random_neighbour(p: Problem, sol: Solution, rng: random.Random) -> Solution:
    """Um movimento aleatório (amostragem barata para o Simulated Annealing)."""
    n = copy.deepcopy(sol)
    where = [(v, k, i) for v in n for k, t in enumerate(n[v]) for i in range(len(t))]
    move = rng.random()
    if move < 0.5:                                               # relocate
        v, k, i = rng.choice(where)
        r = n[v][k].pop(i)
        kind = p.requests[r].kind
        targets = [(v2, k2) for v2 in n for k2, t in enumerate(n[v2]) if not t or _kind(p, t) == kind]
        if targets and rng.random() < 0.85:
            v2, k2 = rng.choice(targets)
            n[v2][k2].insert(rng.randint(0, len(n[v2][k2])), r)
        else:
            v2 = rng.choice(list(n))
            n[v2].insert(rng.randint(0, len(n[v2])), [r])
    elif move < 0.8:                                             # swap
        (va, ka, ia), (vb, kb, ib) = rng.sample(where, 2) if len(where) > 1 else (where[0], where[0])
        if p.requests[n[va][ka][ia]].kind == p.requests[n[vb][kb][ib]].kind:
            n[va][ka][ia], n[vb][kb][ib] = n[vb][kb][ib], n[va][ka][ia]
    else:                                                        # 2-opt
        trips = [(v, k) for v in n for k, t in enumerate(n[v]) if len(t) > 2]
        if trips:
            v, k = rng.choice(trips)
            i, j = sorted(rng.sample(range(len(n[v][k]) + 1), 2))
            n[v][k][i:j] = reversed(n[v][k][i:j])
    return _clean(n)


@dataclass
class LocalResult:
    algorithm: str
    solution: Solution
    initial_cost: float
    cost: float
    feasible: bool
    history: list[float] = field(default_factory=list)      # custo da solução atual em cada iteração
    best_history: list[float] = field(default_factory=list)
    iterations: int = 0
    evaluated: int = 0
    seconds: float = 0.0
    moves: list = field(default_factory=list)

    def summary(self) -> dict:
        return dict(algorithm=self.algorithm, initial_km=round(self.initial_cost, 2), final_km=round(self.cost, 2),
                    improvement_pct=round(100 * (1 - self.cost / self.initial_cost), 2), feasible=self.feasible,
                    iterations=self.iterations, evaluated=self.evaluated, seconds=round(self.seconds, 3))


def hill_climbing(p: Problem, start: Solution, mode: str = "steepest", max_iter: int = 1000,
                  seed: int = 0) -> LocalResult:
    """mode='steepest': melhor vizinho da vizinhança inteira;
    mode='first'   : primeiro vizinho que melhora, com a vizinhança percorrida por ordem aleatória."""
    t0, rng = _time.perf_counter(), random.Random(seed)
    cur = _clean(copy.deepcopy(start))
    ev0 = evaluate(p, cur)
    cur_cost, keep_feasible = ev0.penalized, ev0.feasible
    res = LocalResult(f"HC-{mode}", cur, cur_cost, cur_cost, False, [cur_cost], [cur_cost])
    for _ in range(max_iter):
        best, best_cost, best_move = None, cur_cost, None
        nbrs = list(neighbours(p, cur))
        if mode == "first":
            rng.shuffle(nbrs)
        for move, n in nbrs:
            res.evaluated += 1
            ev = evaluate(p, n)
            if keep_feasible and not ev.feasible:
                continue
            c = ev.penalized
            if c < best_cost - 1e-9:
                best, best_cost, best_move = n, c, move
                if mode == "first":
                    break
        if best is None:
            break                                                # ótimo local
        cur, cur_cost = best, best_cost
        res.iterations += 1
        res.history.append(cur_cost)
        res.best_history.append(cur_cost)
        res.moves.append(best_move)
    ev = evaluate(p, cur)
    res.solution, res.cost, res.feasible, res.seconds = cur, ev.km, ev.feasible, _time.perf_counter() - t0
    return res


def simulated_annealing(p: Problem, start: Solution, t0: float = 50.0, alpha: float = 0.995,
                        iters: int = 20_000, seed: int = 0) -> LocalResult:
    """Aceita vizinhos piores com probabilidade exp(-delta / T); T <- alpha * T."""
    t_start, rng = _time.perf_counter(), random.Random(seed)
    cur = _clean(copy.deepcopy(start))
    ev0 = evaluate(p, cur)
    cur_cost = ev0.penalized
    best, best_cost = cur, (ev0.km if ev0.feasible else float("inf"))     # melhor solução VIÁVEL
    res = LocalResult("SA", cur, cur_cost, cur_cost, False, [cur_cost], [cur_cost])
    T = t0
    for _ in range(iters):
        n = random_neighbour(p, cur, rng)
        ev = evaluate(p, n)
        c = ev.penalized
        res.evaluated += 1
        if c <= cur_cost or rng.random() < math.exp(-(c - cur_cost) / max(T, 1e-9)):
            cur, cur_cost = n, c
            if ev.feasible and ev.km < best_cost - 1e-9:
                best, best_cost = n, ev.km
        T *= alpha
        res.iterations += 1
        res.history.append(cur_cost)
        res.best_history.append(best_cost)
    ev = evaluate(p, best)
    res.solution, res.cost, res.feasible, res.seconds = best, ev.km, ev.feasible, _time.perf_counter() - t_start
    return res

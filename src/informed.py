"""Pesquisa informada (Greedy best-first e A*) sobre soluções parciais de rotas.

Um nó é um :class:`State` (solução parcial). Cada ação estende UM veículo:
  * visitar um pedido por servir (abrindo uma viagem se o veículo estiver parado no armazém), ou
  * fechar a viagem em curso (voltar ao armazém).
Os filhos que violam alguma restrição forte (janela, capacidade, permanência, precedência) são
descartados, e os nós em que algum pedido já não pode ser alcançado a tempo são podados.
Custo do caminho g = km percorridos; as heurísticas são limites inferiores admissíveis dos km que faltam.
"""
from __future__ import annotations

import heapq
import itertools
import time as _time
from dataclasses import dataclass
from typing import Callable

from .model import DAY, DELIVERY, PICKUP, Problem, Solution, State, VState, simulate_trip

Heuristic = Callable[[Problem, State], float]


# --------------------------------------------------------------------------- sucessores
def successors(p: Problem, s: State):
    arrival = dict(s.arrival_w)
    unserved = [r for r in p.requests if r not in s.served]
    for i, v in enumerate(s.vehicles):
        # 1) fechar a viagem em curso
        if v.kind is not None:
            ts = simulate_trip(p, v.vid, list(v.trip), v.trip_free, arrival)
            nv = VState(v.vid, p.depot, ts.end, done=v.done + (v.trip,))
            arr = s.arrival_w
            if v.kind == PICKUP:
                arr = tuple(sorted(arr + tuple((r, ts.back) for r in v.trip)))
            g = s.g - v.trip_km + ts.km
            yield State(s.vehicles[:i] + (nv,) + s.vehicles[i + 1:], s.served, arr, g)
        # 2) visitar um pedido por servir
        for r in unserved:
            q = p.requests[r]
            if v.kind not in (None, q.kind):
                continue                                    # as viagens são homogéneas
            if q.kind == DELIVERY and not all(d in arrival for d in q.deps):
                continue                                    # os produtos ainda não estão no armazém
            free = v.trip_free if v.kind else v.time
            ts = simulate_trip(p, v.vid, list(v.trip) + [r], free, arrival)
            if ts.late > 0 or ts.over > 0 or ts.dwell > 0 or ts.overnight:
                continue
            km = ts.km - p.d(q.loc, p.depot)
            nv = VState(v.vid, q.loc, ts.stops[-1][2] + p.service, q.kind, v.trip + (r,), free, km, v.done)
            yield State(s.vehicles[:i] + (nv,) + s.vehicles[i + 1:], s.served | {r}, s.arrival_w, s.g - v.trip_km + km)


def dead_end(p: Problem, s: State) -> bool:
    """True se algum pedido por servir já não puder começar dentro da janela com nenhum veículo
    (limite otimista: viagem direta, sem esperas, sem problemas de capacidade). Nas entregas usa
    também o primeiro momento em que as suas recolhas podem estar de volta ao armazém."""
    W = p.depot
    known = dict(s.arrival_w)                                   # viagens de recolha fechadas
    at_w = dict(known)                                          # limite inferior da chegada ao armazém
    for v in s.vehicles:
        if v.kind == PICKUP:
            for r in v.trip:
                at_w[r] = v.time + p.t(v.loc, W)

    def earliest(q):
        best = float("inf")
        for v in s.vehicles:
            if v.kind == q.kind:
                t = v.time + p.t(v.loc, q.loc)
            else:
                back = v.time if v.kind is None else v.time + p.t(v.loc, W) + (p.handling if v.kind == PICKUP else 0)
                t = back + (p.handling if q.kind == DELIVERY else 0) + p.t(W, q.loc)
            best = min(best, t)
        return max(best, q.tw[0])

    pending = [r for r in p.requests if r not in s.served]
    for r in pending:                                           # recolhas primeiro: as entregas dependem delas
        q = p.requests[r]
        if q.kind == PICKUP:
            t = earliest(q)
            if t > q.tw[1] + 1e-9:
                return True
            at_w[r] = t + p.service + p.t(q.loc, W)
    for r in pending:
        q = p.requests[r]
        if q.kind == DELIVERY:
            ready = max(at_w[d] + p.handling for d in q.deps)
            t = max(earliest(q), ready + p.handling + p.t(W, q.loc))
            if t > q.tw[1] + 1e-9:
                return True
            closed = [known[d] for d in q.deps if d in known]
            if closed:
                # limite inferior da hora de carga da viagem de q: os produtos têm de estar prontos,
                # e a viagem carrega just in time para a paragem que a abre - uma entrega pendente do
                # mesmo dia ou a primeira paragem de uma viagem de entrega em curso (não
                # necessariamente o próprio q)
                day = q.tw[0] // DAY
                firsts = [o.tw[0] - p.t(W, o.loc) - p.handling for o in (p.requests[x] for x in pending)
                          if o.kind == DELIVERY and o.tw[0] // DAY == day]
                firsts += [f.tw[0] - p.t(W, f.loc) - p.handling for f in
                           (p.requests[v.trip[0]] for v in s.vehicles if v.kind == DELIVERY) if f.tw[0] // DAY == day]
                if max(ready, min(firsts)) - min(closed) > p.max_dwell + 1e-9:
                    return True                                 # não consegue carregar antes do limite de permanência
    return False


# --------------------------------------------------------------------------- heurísticas
def h_max_detour(p: Problem, s: State) -> float:
    """h1: o pedido por servir mais distante ainda tem de ser alcançado por algum veículo e depois
    trazido para o armazém: max_r min_v [d(v, r) + d(r, W)] (desigualdade triangular)."""
    W = p.depot
    out = [v for v in s.vehicles if v.kind is not None]
    h = max((p.d(v.loc, W) for v in out), default=0.0)
    for r, q in p.requests.items():
        if r in s.served:
            continue
        reach = min(p.d(v.loc, q.loc) if v.kind in (None, q.kind) else p.d(v.loc, W) + p.d(W, q.loc)
                    for v in s.vehicles)
        h = max(h, reach + p.d(q.loc, W))
    return h


def _group(p: Problem, rid: str) -> tuple[int, str]:
    """(dia, tipo) de um pedido: uma viagem nunca passa a meia-noite e é homogénea, por isso só
    pedidos do mesmo grupo podem seguir-se dentro de uma viagem."""
    q = p.requests[rid]
    return int(q.tw[0] // DAY), q.kind


def _groups(p: Problem, s: State):
    """Para cada grupo (dia, tipo) ainda ativo: locais por servir, locais dos veículos com viagem em
    curso e número de regressos ao armazém que ainda faltam (viagens em curso + viagens extra pelo peso)."""
    cap = max(v.capacity for v in p.vehicles)
    nodes: dict[tuple[int, str], set[int]] = {}
    weight: dict[tuple[int, str], float] = {}
    for r, q in p.requests.items():
        if r not in s.served:
            g = _group(p, r)
            nodes.setdefault(g, set()).add(q.loc)
            weight[g] = weight.get(g, 0.0) + q.weight
    opened: dict[tuple[int, str], list[VState]] = {}
    for v in s.vehicles:
        if v.kind is not None:
            opened.setdefault(_group(p, v.trip[0]), []).append(v)
    for g in sorted(nodes.keys() | opened.keys()):         # ordem fixa: h é uma soma em vírgula flutuante
        vs = opened.get(g, [])
        spare = sum(p.vehicle(v.vid).capacity - sum(p.requests[r].weight for r in v.trip) for v in vs)
        new_trips = max(0, -(-(weight.get(g, 0.0) - spare) // cap)) if g in nodes else 0
        if g in nodes and not vs:
            pending = [r for r in p.requests if r not in s.served and _group(p, r) == g]
            new_trips = max(new_trips, 1, _clique(p, pending))
        yield g, nodes.get(g, set()), [v.loc for v in vs], len(vs) + int(new_trips)


def _incompatible(p: Problem) -> set[frozenset]:
    """Pares de pedidos que nunca podem partilhar viagem: janelas, peso ou - nas entregas - o
    primeiro momento em que os produtos podem estar no armazém (uma viagem só carrega quando estão todos)."""
    if "incompatible" not in p.cache:
        cap, bad, W = max(v.capacity for v in p.vehicles), set(), p.depot
        ready = {r: max((p.requests[d].tw[0] + p.service + p.t(p.requests[d].loc, W) + p.handling
                         for d in q.deps), default=0.0) for r, q in p.requests.items()}
        for a, qa in p.requests.items():
            for b, qb in p.requests.items():
                if a < b and _group(p, a) == _group(p, b):
                    ab = qa.tw[0] + p.service + p.t(qa.loc, qb.loc) <= qb.tw[1]
                    ba = qb.tw[0] + p.service + p.t(qb.loc, qa.loc) <= qa.tw[1]
                    late = (qa.kind == DELIVERY and
                            (ready[b] + p.handling + p.t(W, qa.loc) > qa.tw[1]
                             or ready[a] + p.handling + p.t(W, qb.loc) > qb.tw[1]))
                    if not (ab or ba) or late or qa.weight + qb.weight > cap:
                        bad.add(frozenset((a, b)))
        p.cache["incompatible"] = bad
    return p.cache["incompatible"]


def _clique(p: Problem, rids: list[str]) -> int:
    """Tamanho de uma clique (gulosa) de pedidos incompatíveis entre si: cada um precisa da sua viagem."""
    bad = _incompatible(p)
    deg = {r: sum(frozenset((r, o)) in bad for o in rids if o != r) for r in rids}
    clique: list[str] = []
    for r in sorted(rids, key=lambda x: -deg[x]):
        if all(frozenset((r, c)) in bad for c in clique):
            clique.append(r)
    return len(clique)


def _min_return(p: Problem, locs, vlocs) -> float:
    return min([p.d(b, p.depot) for b in locs] + [p.d(a, p.depot) for a in vlocs])


def h_min_arcs(p: Problem, s: State) -> float:
    """h2: relaxação sobre nós distintos (local, dia, tipo).
    * cada nó por servir precisa de um arco de entrada, vindo do armazém, de outro nó do mesmo
      grupo (dia, tipo) ou de um veículo cuja viagem em curso pertence a esse grupo;
    * cada viagem ainda em curso ou ainda necessária acaba com um arco de regresso ao armazém.
    Cada arco contado é distinto em qualquer completamento, por isso a soma dos mínimos é admissível."""
    W, h = p.depot, 0.0
    for g, locs, vlocs, n_ret in _groups(p, s):
        for loc in locs:
            if loc not in vlocs:                           # uma viagem em curso que já está nesse local custa 0
                h += min([p.d(W, loc)] + [p.d(a, loc) for a in vlocs] + [p.d(b, loc) for b in locs if b != loc])
        h += n_ret * _min_return(p, locs, vlocs)
    return h


def h_mst(p: Problem, s: State) -> float:
    """h3: em cada grupo (dia, tipo), tirar o arco de regresso de cada viagem que falta deixa uma
    floresta com raiz no armazém (os veículos em curso ligam-se a ela com custo 0) que cobre todos
    os locais por servir -> o seu custo é >= a árvore de cobertura mínima. Somam-se os arcos de
    regresso mais baratos."""
    return sum(_mst_bounds(p, s).values())


def _mst_bounds(p: Problem, s: State) -> dict[tuple[int, str], float]:
    """O limite h3 de cada grupo (dia, tipo) ativo."""
    W, out = p.depot, {}
    for g, locs, vlocs, n_ret in _groups(p, s):
        bound = _prim(p, {W, *vlocs}, set(locs)) + n_ret * _min_return(p, locs, vlocs)
        if locs and not vlocs:
            # limite do tipo 1-tree: com m viagens, tirar o armazém deixa no máximo m componentes
            # que cobrem todos os locais (>= MST menos as suas m-1 arestas mais longas), e o armazém
            # tem 2m arestas incidentes, cada uma >= ao local mais próximo. Os locais podem ser
            # revisitados por várias viagens, por isso não se assume nada mais forte para essas 2m arestas.
            edges = sorted(_prim_edges(p, set(locs)), reverse=True)
            tree, w_min = sum(edges), min(p.d(W, b) for b in locs)
            lo = max(n_ret, 1)
            one_tree = min(tree - sum(edges[:m - 1]) + 2 * m * w_min for m in range(lo, max(lo, len(locs)) + 1))
            bound = max(bound, one_tree)
        out[g] = bound
    return out


class _RelaxedVRP:
    """Solução exata de uma relaxação, calculada uma vez por problema e por grupo (dia, tipo).

    Relaxação: cada grupo é servido sozinho por uma frota ilimitada do maior veículo, e cada viagem
    sai do armazém o mais cedo possível (entregas: assim que o regresso mais cedo possível das suas
    recolhas foi descarregado e a viagem carregada). Mantêm-se as janelas, os tempos de serviço e
    de manuseamento, a capacidade e a regra da meia-noite. Largam-se a partilha de veículos entre
    grupos, as horas exatas das recolhas e o limite de permanência, por isso o ótimo da relaxação é
    um limite inferior.

    1. Viagens viáveis: etiquetas sobre (subconjunto, último pedido) com a hora de serviço mais
       cedo (exata para a viabilidade, porque esperar é permitido) e os menores km (um limite
       inferior dos km de qualquer ordem viável). Nas entregas, uma passagem por "nível de
       prontidão" fixa a hora de partida das viagens cujo pedido pronto mais tarde tem esse nível.
    2. f(U) = partição de custo mínimo do conjunto por servir U em viagens viáveis (PD com memória).
    """

    def __init__(self, p: Problem):
        self.p, W = p, p.depot
        cap = max(v.capacity for v in p.vehicles)
        ready = {r: max((p.requests[d].tw[0] + p.service + p.t(p.requests[d].loc, W) + p.handling for d in q.deps),
                        default=0.0) for r, q in p.requests.items()}
        self.groups: dict[tuple[int, str], list[str]] = {}
        for r in p.requests:
            self.groups.setdefault(_group(p, r), []).append(r)
        self.index = {r: (g, i) for g, rs in self.groups.items() for i, r in enumerate(rs)}
        self.by_low: dict[tuple[int, str], dict[int, list[tuple[int, float]]]] = {}
        self.memo: dict[tuple[int, str], dict[int, float]] = {}
        for g, R in self.groups.items():
            day, kind = g
            Q = [p.requests[r] for r in R]
            levels = sorted({ready[r] for r in R}) if kind == DELIVERY else [day * DAY]
            trips: dict[int, float] = {}
            for L in levels:
                dep = max(L, day * DAY) + (p.handling if kind == DELIVERY else 0.0)
                allowed = [i for i in range(len(R)) if kind == PICKUP or ready[R[i]] <= L]
                lab: dict[tuple[int, int], list[float]] = {}
                frontier = []
                for i in allowed:
                    st = max(dep + p.t(W, Q[i].loc), Q[i].tw[0])
                    if st <= Q[i].tw[1] and Q[i].weight <= cap:
                        lab[1 << i, i] = [st + p.service, p.d(W, Q[i].loc), Q[i].weight]
                        frontier.append((1 << i, i))
                while frontier:                                   # em largura, pelo tamanho do subconjunto
                    nxt = []
                    for m, j in frontier:
                        tj, kj, wj = lab[m, j]
                        for k in allowed:
                            if m >> k & 1 or wj + Q[k].weight > cap:
                                continue
                            st = max(tj + p.t(Q[j].loc, Q[k].loc), Q[k].tw[0])
                            if st > Q[k].tw[1]:
                                continue
                            key, t_new, k_new = (m | 1 << k, k), st + p.service, kj + p.d(Q[j].loc, Q[k].loc)
                            if key in lab:
                                o = lab[key]
                                o[0], o[1] = min(o[0], t_new), min(o[1], k_new)
                            else:
                                lab[key] = [t_new, k_new, wj + Q[k].weight]
                                nxt.append(key)
                    frontier = nxt
                for (m, j), (tj, kj, _) in lab.items():
                    if tj + p.t(Q[j].loc, W) > (day + 1) * DAY:
                        continue                                  # não voltaria antes da meia-noite
                    if kind == DELIVERY and max(ready[R[i]] for i in range(len(R)) if m >> i & 1) != L:
                        continue                                  # pertence a outro nível de prontidão
                    trips[m] = min(trips.get(m, float("inf")), kj + p.d(Q[j].loc, W))
            low: dict[int, list[tuple[int, float]]] = {}
            for m, c in trips.items():
                low.setdefault((m & -m).bit_length() - 1, []).append((m, c))
            for lst in low.values():
                lst.sort(key=lambda x: x[1])
            self.by_low[g], self.memo[g] = low, {0: 0.0}

    def f(self, g, U: int) -> float:
        """Menor número de km para servir os pedidos da máscara U do grupo g (inf se for impossível)."""
        memo = self.memo[g]
        if U in memo:
            return memo[U]
        stack, order = [U], []
        while stack:                                              # DFS iterativa (sem limite de recursão)
            V = stack.pop()
            if V in memo:
                continue
            low = (V & -V).bit_length() - 1
            missing = [V & ~m for m, _ in self.by_low[g].get(low, ()) if m & ~V == 0 and (V & ~m) not in memo]
            if missing:
                stack.append(V)
                stack.extend(missing)
            else:
                memo[V] = min((c + memo[V & ~m] for m, c in self.by_low[g].get(low, ()) if m & ~V == 0),
                              default=float("inf"))
        return memo[U]


def h_relaxed(p: Problem, s: State) -> float:
    """h4: em cada grupo (dia, tipo), o melhor entre a h3 e o ótimo exato de um problema de rotas
    relaxado (ver :class:`_RelaxedVRP`). Com viagens em curso, o ótimo relaxado é reduzido em
    d(W, veículo) por cada uma: acabar uma viagem em curso por um conjunto S custa pelo menos uma
    viagem nova por S menos essa distância (desigualdade triangular)."""
    rv = _relaxed(p)
    mask: dict[tuple[int, str], int] = {}
    for r in p.requests:
        if r not in s.served:
            g, i = rv.index[r]
            mask[g] = mask.get(g, 0) | 1 << i
    credit: dict[tuple[int, str], float] = {}
    for v in s.vehicles:
        if v.kind is not None:
            g = _group(p, v.trip[0])
            credit[g] = credit.get(g, 0.0) + p.d(p.depot, v.loc)
    h = 0.0
    for g, b3 in _mst_bounds(p, s).items():
        relaxed = rv.f(g, mask[g]) - credit.get(g, 0.0) if g in mask else 0.0
        h += max(b3, relaxed)
    return h


def _prim_edges(p: Problem, nodes: set[int]) -> list[float]:
    """Comprimentos das arestas de uma árvore de cobertura mínima sobre ``nodes``."""
    if len(nodes) < 2:
        return []
    first, *rest = nodes
    best = {b: p.d(first, b) for b in rest}
    out = []
    while best:
        b = min(best, key=best.get)
        out.append(best.pop(b))
        for c in best:
            best[c] = min(best[c], p.d(b, c))
    return out


def _prim(p: Problem, roots: set[int], nodes: set[int]) -> float:
    """Custo da MST que liga ``nodes`` às ``roots`` (já ligadas entre si)."""
    best = {b: min(p.d(a, b) for a in roots) for b in nodes - roots}
    total = 0.0
    while best:
        b = min(best, key=best.get)
        total += best.pop(b)
        for c in best:
            best[c] = min(best[c], p.d(b, c))
    return total


def h_zero(p: Problem, s: State) -> float:
    """Sem heurística: o A* passa a pesquisa de custo uniforme (usada nos testes)."""
    return 0.0


HEURISTICS = {"h1_max_detour": h_max_detour, "h2_min_arcs": h_min_arcs, "h3_mst": h_mst, "h4_relaxed": h_relaxed}


# --------------------------------------------------------------------------- pesquisa
@dataclass
class SearchResult:
    algorithm: str
    heuristic: str
    solution: Solution | None
    cost: float | None
    expanded: int
    generated: int
    pruned: int
    seconds: float
    complete: bool              # False se parou pelo limite de expansões

    def summary(self) -> dict:
        return dict(algorithm=self.algorithm, heuristic=self.heuristic,
                    cost_km=None if self.cost is None else round(self.cost, 2), expanded=self.expanded,
                    generated=self.generated, pruned=self.pruned, seconds=round(self.seconds, 3),
                    found=self.solution is not None, complete=self.complete)


def latest_loads(p: Problem) -> dict[str, float]:
    """Para cada recolha, o último momento em que alguma entrega que precisa dela pode começar a carregar."""
    out: dict[str, float] = {}
    for q in p.requests.values():
        for d in q.deps:
            out[d] = max(out.get(d, -1e18), q.tw[1] - p.t(p.depot, q.loc) - p.handling)
    return out


def dwell_safe(p: Problem, s: State, latest: dict[str, float]) -> bool:
    """True se nenhuma entrega pendente de s puder alguma vez violar o limite de permanência, seja
    qual for o horário: nesse caso chegar mais cedo ao armazém só ajuda, e a dominância <= sobre as
    chegadas é exata."""
    return all(latest.get(r, -1e18) - t <= p.max_dwell + 1e-9 for r, t in s.arrival_w)


def _dominated(store: dict, key, g: float, times: tuple) -> bool:
    """s é dominado por um estado guardado com a mesma chave e custo e relógios menores ou iguais.
    Os estados guardados só servem para isto quando são dwell-safe (flag), o que torna a poda
    exata (nunca se descarta uma solução ótima)."""
    for g0, t0, safe in store.get(key, ()):
        if g0 <= g + 1e-9 and (all(a <= b + 1e-9 for a, b in zip(t0, times)) if safe else t0 == times):
            return True
    return False


def chronological(p: Problem, s: State, children: list[State]) -> list[State]:
    """Fica só com os movimentos do veículo com o relógio mais atrasado entre os que se podem mover.
    O A* não a usa (podia cortar o ótimo); serve para o Greedy não mandar um veículo para o dia
    seguinte enquanto o outro ainda tem trabalho pendente."""
    by_vehicle: dict[int, list[State]] = {}
    for c in children:
        i = next(k for k, (a, b) in enumerate(zip(s.vehicles, c.vehicles)) if a is not b)
        by_vehicle.setdefault(i, []).append(c)
    if not by_vehicle:
        return []
    first = min(by_vehicle, key=lambda i: (s.vehicles[i].time, i))
    return by_vehicle[first]


def best_first(p: Problem, h: Heuristic, w_g: float, w_h: float, name: str,
               max_expansions: int = 200_000, chrono: bool = False, succ=None,
               time_limit: float = float("inf")) -> SearchResult:
    """Pesquisa best-first genérica com f = w_g * g + w_h * h.
    A* -> (1, 1); Greedy best-first -> (0, 1); A* ponderado -> (1, w>1).
    chrono=True limita cada expansão ao veículo mais atrasado (ver :func:`chronological`).
    succ: função de sucessores (por omissão: :func:`successors`, paragem a paragem).
    A pesquisa pára (sem prova) ao fim de max_expansions ou de time_limit segundos."""
    succ = succ or successors
    t0 = _time.perf_counter()
    tie = itertools.count()
    start = State.initial(p)
    frontier = [(w_h * h(p, start), 0, next(tie), start)]
    store: dict = {}
    latest = latest_loads(p)
    expanded = generated = pruned = 0
    while frontier:
        _, _, _, s = heapq.heappop(frontier)
        if s.is_goal(p):
            return SearchResult(name, h.__name__, s.solution(), s.g, expanded, generated, pruned,
                                _time.perf_counter() - t0, True)
        if expanded >= max_expansions or _time.perf_counter() - t0 > time_limit:
            frontier.append(None)                           # parou pelo orçamento: não é exaustiva
            break
        expanded += 1
        children = list(succ(p, s))
        if chrono:
            children = chronological(p, s, children)
        for c in children:
            generated += 1
            key, times = c.key(), c.times()
            if _dominated(store, key, c.g, times) or dead_end(p, c):
                pruned += 1
                continue
            store.setdefault(key, []).append((c.g, times, dwell_safe(p, c, latest)))
            hc = h(p, c)
            heapq.heappush(frontier, (w_g * c.g + w_h * hc, -len(c.served), next(tie), c))
    return SearchResult(name, h.__name__, None, None, expanded, generated, pruned,
                        _time.perf_counter() - t0, not frontier)


def astar(p: Problem, h: Heuristic = h_min_arcs, **kw) -> SearchResult:
    return best_first(p, h, 1.0, 1.0, "A*", **kw)


def greedy(p: Problem, h: Heuristic = h_min_arcs, chrono: bool = True, **kw) -> SearchResult:
    return best_first(p, h, 0.0, 1.0, "Greedy", chrono=chrono, **kw)


def weighted_astar(p: Problem, h: Heuristic = h_min_arcs, w: float = 2.0, **kw) -> SearchResult:
    return best_first(p, h, 1.0, w, f"WA*(w={w})", **kw)


# --------------------------------------------------------------------------- pesquisa por viagens
@dataclass(frozen=True)
class TripOption:
    """Uma forma de fazer uma viagem: que pedidos, por que ordem, e como o horário depende da hora
    de partida D do armazém (forma max-plus, esperas permitidas):
        back(D) = max(D + delta, c)            viável  <=>  D <= d_max
    jit: partida mais cedo que faz sentido (chegar à primeira paragem quando a janela abre)."""
    group: tuple[int, str]
    mask: int
    rids: tuple[str, ...]
    km: float
    delta: float
    c: float
    d_max: float
    jit: float
    weight: float
    deps: frozenset


class TripCatalog:
    """Todas as viagens que podem fazer parte de uma solução, sem as ordens dominadas.

    Em cada grupo (dia, tipo), as ordens são enumeradas em profundidade a partir da partida mais
    cedo possível (recolhas: início do dia; entregas: primeiro momento em que todos os produtos
    podem estar no armazém, mais a carga). Uma partida real mais tarde só pode tornar uma ordem
    inviável, por isso isto é um superconjunto de todas as viagens de todas as soluções viáveis.
    Duas ordens do mesmo conjunto são comparadas em (km, delta, c, d_max, jit): se uma for pelo
    menos tão boa nos cinco, é pelo menos tão boa para QUALQUER hora de partida, qualquer viagem
    seguinte e o limite de permanência, por isso a outra é largada sem perder o ótimo (redução exata)."""

    def __init__(self, p: Problem):
        self.p, W = p, p.depot
        cap = max(v.capacity for v in p.vehicles)
        rv = _relaxed(p)
        self.index = rv.index
        ready = {r: max((p.requests[d].tw[0] + p.service + p.t(p.requests[d].loc, W) + p.handling
                         for d in q.deps), default=0.0) for r, q in p.requests.items()}
        self.options: list[TripOption] = []
        self.raw_orders = 0
        for g, R in rv.groups.items():
            day, kind = g
            end_of_day = (day + 1) * DAY
            best: dict[int, list[TripOption]] = {}

            def dfs(seq, mask, weight, ready_lvl):
                for i, r in enumerate(R):
                    if mask >> i & 1:
                        continue
                    q = p.requests[r]
                    if weight + q.weight > cap:
                        continue
                    lvl = max(ready_lvl, ready[r]) if kind == DELIVERY else ready_lvl
                    order = seq + (r,)
                    opt = self._option(g, order, mask | 1 << i, weight + q.weight)
                    dep0 = max(lvl, day * DAY) + (p.handling if kind == DELIVERY else 0.0)
                    if dep0 > opt.d_max or max(dep0 + opt.delta, opt.c) >= end_of_day:
                        continue                          # inviável agora -> inviável também mais tarde
                    self.raw_orders += 1
                    lst = best.setdefault(opt.mask, [])
                    if not any(_dominates(o, opt) for o in lst):
                        lst[:] = [o for o in lst if not _dominates(opt, o)] + [opt]
                    dfs(order, mask | 1 << i, weight + q.weight, lvl)

            dfs((), 0, 0.0, 0.0)
            for lst in best.values():
                self.options.extend(lst)
        self.options.sort(key=lambda o: o.km)

    def _option(self, g, order, mask, weight) -> TripOption:
        p, W = self.p, self.p.depot
        qs = [p.requests[r] for r in order]
        # delta: duração sem esperas; c: hora de regresso ao sair "muito cedo".
        # Cada início de serviço é max(D + delta_k, c_k); tem de ser <= ao fim da janela, o que exige
        # D <= tw_end - delta_k (incluído em d_max abaixo) E c_k <= tw_end (espera imposta por
        # janelas anteriores) - esta última não depende de D, por isso a ordem fica simplesmente inviável.
        delta, t, loc, forced_late = 0.0, -1e9, W, False
        for q in qs:
            delta += p.t(loc, q.loc)
            t = max(t + p.t(loc, q.loc), q.tw[0])
            forced_late |= t > q.tw[1] + 1e-9
            delta += p.service
            t += p.service
            loc = q.loc
        delta += p.t(loc, W)
        c = t + p.t(loc, W)
        # d_max: partida mais tarde que ainda começa cada paragem dentro da janela (de trás para a frente)
        latest, nxt = float("inf"), None
        for q in reversed(qs):
            lt = q.tw[1] if nxt is None else min(q.tw[1], latest - p.service - p.t(q.loc, nxt.loc))
            latest, nxt = lt, q
        d_max = -float("inf") if forced_late else latest - p.t(W, qs[0].loc)
        km = p.d(W, qs[0].loc) + sum(p.d(a.loc, b.loc) for a, b in zip(qs, qs[1:])) + p.d(qs[-1].loc, W)
        jit = qs[0].tw[0] - p.t(W, qs[0].loc)
        deps = frozenset(d for q in qs for d in q.deps)
        return TripOption(g, mask, tuple(order), km, delta, c, d_max, jit, weight, deps)


def _dominates(a: TripOption, b: TripOption) -> bool:
    e = 1e-9
    return (a.km <= b.km + e and a.delta <= b.delta + e and a.c <= b.c + e and a.d_max >= b.d_max - e
            and a.jit <= b.jit + e)


def _relaxed(p: Problem) -> _RelaxedVRP:
    if "relaxed" not in p.cache:
        p.cache["relaxed"] = _RelaxedVRP(p)
    return p.cache["relaxed"]


def catalog(p: Problem) -> TripCatalog:
    if "catalog" not in p.cache:
        p.cache["catalog"] = TripCatalog(p)
    return p.cache["catalog"]


def trip_successors(p: Problem, s: State):
    """Ações ao nível da viagem: um veículo parado faz uma viagem completa do catálogo.
    O horário é calculado de forma exata a partir de (delta, c, d_max, jit) da viagem e do contexto
    real (hora a que o veículo fica livre, chegadas reais das recolhas ao armazém), com as mesmas
    regras de :func:`simulate_trip`: janelas, capacidade, nenhuma viagem passa a meia-noite,
    permanência <= max_dwell."""
    cat = catalog(p)
    arrival = dict(s.arrival_w)
    served_mask: dict = {}
    for r in s.served:
        g, i = cat.index[r]
        served_mask[g] = served_mask.get(g, 0) | 1 << i
    for i, v in enumerate(s.vehicles):
        cap = p.vehicle(v.vid).capacity
        for o in cat.options:
            if o.mask & served_mask.get(o.group, 0) or o.weight > cap + 1e-9:
                continue
            if o.group[1] == DELIVERY:
                if not o.deps <= arrival.keys():
                    continue
                ready = max(arrival[d] + p.handling for d in o.deps)
                load_start = max(v.time, ready, o.jit - p.handling)
                if any(load_start - arrival[d] > p.max_dwell + 1e-9 for d in o.deps):
                    continue
                dep = load_start + p.handling
            else:
                load_start = dep = max(v.time, o.jit)
            if dep > o.d_max + 1e-9:
                continue
            back = max(dep + o.delta, o.c)
            if back // DAY != load_start // DAY:
                continue                                    # nenhuma viagem passa a meia-noite
            end = back + p.handling if o.group[1] == PICKUP else back
            nv = VState(v.vid, p.depot, end, done=v.done + (o.rids,))
            arr = s.arrival_w
            if o.group[1] == PICKUP:
                arr = tuple(sorted(arr + tuple((r, back) for r in o.rids)))
            yield State(s.vehicles[:i] + (nv,) + s.vehicles[i + 1:], s.served | set(o.rids), arr, s.g + o.km)


def astar_trips(p: Problem, h: Heuristic = None, **kw) -> SearchResult:
    """A* em que cada ação é uma viagem inteira (mesmo estado, custo, objetivo e heurísticas)."""
    return best_first(p, h or h_relaxed, 1.0, 1.0, "A*-viagens", succ=trip_successors, **kw)

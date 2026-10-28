"""Formulação CSP do problema de rotas.

Variáveis  : uma por pedido r (grupo de recolha ou entrega de uma encomenda).
Domínio    : (veículo, dia, posição). Cada dia tem um modelo fixo de posições de viagem, por
             exemplo D P D D P D D (D = viagem de entrega, P = viagem de recolha), por isso um
             valor fixa o veículo que serve r, em que viagem e em que dia. Os pedidos com o
             mesmo valor formam uma viagem, visitada pela ordem mais curta que cumpre as
             janelas a partir da hora real de saída do veículo (ver ``order_in_context``).
Unárias    : tipo da posição = tipo do pedido; dia da posição = dia do pedido (nenhuma viagem
             passa a meia-noite).
Binárias   : (mesma viagem) capacidade  w_a + w_b <= cap(v);
             (mesma viagem) compatibilidade temporal: a e b podem seguir-se dentro das janelas
                            e, nas entregas, os produtos de um não ficam prontos tarde demais
                            para o outro;
             (precedência)  entrega e uma das suas recolhas no mesmo veículo: a viagem da
                            recolha vem primeiro;
             (sequência)    mesmo veículo, viagem de a antes da de b => b tem de continuar
                            alcançável dentro da janela depois de a viagem de a voltar ao armazém.
Globais    : capacidade de cada viagem (n-ária) e horário completo (janelas, precedência entre
             veículos no armazém, permanência <= 24 h), verificado pelo simulador sobre a
             atribuição parcial.
Suave      : km totais (e menos viagens), tratados pela ordenação de valores e, opcionalmente,
             por branch & bound depois da primeira solução.

Algoritmos: backtracking com MRV + grau opcional, ordenação de valores (inserção mais barata /
least constraining value), forward checking, MAC (AC-3) e, opcionalmente, forward checking
contra a restrição global do horário (``global_fc``).
"""
from __future__ import annotations

import itertools
import time as _time
from collections import deque
from dataclasses import dataclass, field

from .informed import _incompatible
from .model import DAY, DELIVERY, PICKUP, Problem, Solution, evaluate, simulate_trip

TEMPLATE = (DELIVERY, PICKUP, DELIVERY, DELIVERY, PICKUP, DELIVERY, DELIVERY)
Value = tuple[int, int, int]            # (id do veículo, dia, índice da posição)


@dataclass
class CSPResult:
    solution: Solution | None
    cost: float | None
    feasible: bool
    assignments: int = 0
    backtracks: int = 0
    checks: int = 0
    pruned: int = 0
    seconds: float = 0.0
    solutions_found: int = 0
    status: str = ""
    config: dict = field(default_factory=dict)
    improvements: list = field(default_factory=list)   # (atribuições, segundos, km) por cada solução melhor

    def summary(self) -> dict:
        return dict(**self.config, status=self.status, cost_km=None if self.cost is None else round(self.cost, 2),
                    assignments=self.assignments, backtracks=self.backtracks, checks=self.checks,
                    pruned=self.pruned, solutions=self.solutions_found, seconds=round(self.seconds, 3))


class RoutingCSP:
    def __init__(self, p: Problem, template: tuple[str, ...] = TEMPLATE):
        self.p, self.template = p, template
        self.vars = list(p.requests)
        self.day = {r: int(q.tw[0] // DAY) for r, q in p.requests.items()}
        self.domains: dict[str, list[Value]] = {
            r: [(v.vid, self.day[r], k) for v in p.vehicles for k, kind in enumerate(template) if kind == q.kind]
            for r, q in p.requests.items()}
        bad = _incompatible(p)
        self.compat = {(a, b): self._time_compatible(a, b) and frozenset((a, b)) not in bad
                       for a in self.vars for b in self.vars if a != b}
        self.seq_ok = {(a, b): self._can_precede(a, b) for a in self.vars for b in self.vars if a != b}
        # grafo das restrições binárias: pares que podem partilhar viagem, pares entrega-recolha e
        # pares cujas viagens não podem estar numa certa ordem no mesmo veículo
        self.neigh: dict[str, list[str]] = {r: [] for r in self.vars}
        for a in self.vars:
            for b in self.vars:
                if a != b and (self._same_group(a, b) or b in p.requests[a].deps or a in p.requests[b].deps
                               or not self.seq_ok[a, b] or not self.seq_ok[b, a]):
                    self.neigh[a].append(b)
        self._order_cache: dict[frozenset, list[tuple[str, ...]]] = {}
        self._context_cache: dict[tuple, tuple] = {}

    # ---------------------------------------------------------------- restrições
    def _same_group(self, a: str, b: str) -> bool:
        return self.day[a] == self.day[b] and self.p.requests[a].kind == self.p.requests[b].kind

    def _time_compatible(self, a: str, b: str) -> bool:
        """a e b podem ser visitados um a seguir ao outro (numa das ordens) dentro das janelas?"""
        p, qa, qb = self.p, self.p.requests[a], self.p.requests[b]
        ab = qa.tw[0] + p.service + p.t(qa.loc, qb.loc) <= qb.tw[1]
        ba = qb.tw[0] + p.service + p.t(qb.loc, qa.loc) <= qa.tw[1]
        return ab or ba

    def _can_precede(self, a: str, b: str) -> bool:
        """Uma viagem com a pode vir (no mesmo veículo) antes de uma viagem com b?"""
        p, qa, qb, W = self.p, self.p.requests[a], self.p.requests[b], self.p.depot
        back = qa.tw[0] + p.service + p.t(qa.loc, W) + (p.handling if qa.kind == PICKUP else 0)
        return back + (p.handling if qb.kind == DELIVERY else 0) + p.t(W, qb.loc) <= qb.tw[1] + 1e-9

    def order_key(self, val: Value) -> tuple[int, int]:
        return val[1], val[2]

    def consistent(self, a: str, va: Value, b: str, vb: Value) -> bool:
        """Todas as restrições binárias entre a=va e b=vb."""
        p = self.p
        if va == vb:
            if not self.compat[a, b]:
                return False
            if p.requests[a].weight + p.requests[b].weight > p.vehicle(va[0]).capacity + 1e-9:
                return False
        if va[0] == vb[0]:
            ka, kb = self.order_key(va), self.order_key(vb)
            if a in p.requests[b].deps and not ka < kb:
                return False
            if b in p.requests[a].deps and not kb < ka:
                return False
            if ka < kb and not self.seq_ok[a, b]:
                return False
            if kb < ka and not self.seq_ok[b, a]:
                return False
        return True

    def trip_km(self, seq) -> float:
        p, W = self.p, self.p.depot
        return (p.d(W, p.requests[seq[0]].loc) + p.d(p.requests[seq[-1]].loc, W)
                + sum(p.d(p.requests[x].loc, p.requests[y].loc) for x, y in zip(seq, seq[1:])))

    def _two_opt(self, seq: tuple[str, ...]) -> tuple[str, ...]:
        """Encurta uma ordem invertendo segmentos (só olha para os km; as janelas são verificadas depois)."""
        best, improved = list(seq), True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best) + 1):
                    cand = best[:i] + best[i:j][::-1] + best[j:]
                    if self.trip_km(cand) < self.trip_km(best) - 1e-9:
                        best, improved = cand, True
        return tuple(best)

    def candidate_orders(self, members: frozenset) -> list[tuple[str, ...]]:
        """Ordens de visita a experimentar numa viagem, da mais curta para a mais longa."""
        if members in self._order_cache:
            return self._order_cache[members]
        p, W = self.p, self.p.depot
        rs = sorted(members)                    # ordem fixa: empates em km (viagens invertidas) ficam estáveis
        if len(rs) <= 6:
            orders = list(itertools.permutations(rs))
        else:
            # Até 6 pedidos experimentamos todas as permutações (6! = 720). As maiores viagens do
            # small têm 11 pedidos, o que seriam cerca de 40 milhões de permutações a testar em cada
            # verificação do horário. Por isso, acima de 6 experimentamos só algumas ordens razoáveis
            # (por fecho de janela, por abertura de janela e pelo vizinho mais próximo, cada uma
            # também encurtada com 2-opt). É rápido, mas pode falhar a melhor ordem.
            by_close = tuple(sorted(rs, key=lambda r: (p.requests[r].tw[1], r)))
            by_open = tuple(sorted(rs, key=lambda r: (p.requests[r].tw[0], r)))
            nearest, loc, left = [], W, set(rs)
            while left:
                nxt = min(sorted(left), key=lambda r: p.d(loc, p.requests[r].loc))
                nearest.append(nxt)
                left.remove(nxt)
                loc = p.requests[nxt].loc
            seeds = [by_close, by_open, tuple(nearest)]
            orders = list(dict.fromkeys(seeds + [self._two_opt(s) for s in seeds]))
        orders.sort(key=self.trip_km)
        self._order_cache[members] = orders
        return orders

    def order_in_context(self, vid: int, members: frozenset, free: float, arrival_w: dict[str, float]):
        """Ordem candidata mais curta que cumpre as janelas e o limite de permanência quando o
        veículo fica livre em ``free`` e as recolhas chegaram ao armazém em ``arrival_w``.
        Antes, a ordem era escolhida como se o veículo saísse às 00:00, e o CSP rejeitava
        atribuições que outra ordem tornava viáveis. Se nenhuma ordem servir, devolve a mais
        curta (e quem chama vê a violação)."""
        deps = sorted({d for r in members for d in self.p.requests[r].deps})
        key = (vid, members, round(free, 6), tuple(round(arrival_w[d], 6) for d in deps))
        if key in self._context_cache:
            return self._context_cache[key]
        orders = self.candidate_orders(members)
        chosen = None
        for seq in orders:
            ts = simulate_trip(self.p, vid, list(seq), free, arrival_w)
            if ts.late == 0 and ts.dwell == 0 and not ts.overnight:
                chosen = (list(seq), ts)
                break
        if chosen is None:
            chosen = (list(orders[0]), simulate_trip(self.p, vid, list(orders[0]), free, arrival_w))
        self._context_cache[key] = chosen
        return chosen

    def to_solution(self, assign: dict[str, Value]) -> Solution:
        """Viagens de cada veículo pela ordem das posições, cada uma visitada pela ordem que
        ``order_in_context`` escolhe. O horário é construído como em ``evaluate``: uma viagem de
        entrega espera pelas suas recolhas."""
        p = self.p
        trips: dict[Value, set[str]] = {}
        for r, val in assign.items():
            trips.setdefault(val, set()).add(r)
        queue = {v.vid: [frozenset(trips[val]) for val in sorted(trips, key=self.order_key) if val[0] == v.vid]
                 for v in p.vehicles}
        sol: Solution = {vid: [] for vid in queue}
        arrival_w: dict[str, float] = {}
        free = {vid: 0.0 for vid in queue}
        progress = True
        while progress:
            progress = False
            for vid, members_list in queue.items():
                while len(sol[vid]) < len(members_list):
                    members = members_list[len(sol[vid])]
                    if not {d for r in members for d in p.requests[r].deps} <= arrival_w.keys():
                        break                                   # espera por outro veículo
                    seq, ts = self.order_in_context(vid, members, free[vid], arrival_w)
                    sol[vid].append(seq)
                    if ts.kind == PICKUP:
                        for rid, _, _ in ts.stops:
                            arrival_w[rid] = ts.back
                    free[vid] = ts.end
                    progress = True
        for vid, members_list in queue.items():                 # viagens bloqueadas ficam com a ordem mais curta
            sol[vid].extend(list(self.candidate_orders(members)[0]) for members in members_list[len(sol[vid]):])
        return sol

    def global_ok(self, assign: dict[str, Value]) -> tuple[bool, float]:
        """Capacidade de cada viagem + horário completo da atribuição parcial (as entregas cujas
        recolhas ainda não foram atribuídas ficam de fora até o serem)."""
        p = self.p
        load: dict[Value, float] = {}
        for r, val in assign.items():
            load[val] = load.get(val, 0.0) + p.requests[r].weight
            if load[val] > p.vehicle(val[0]).capacity + 1e-9:
                return False, 0.0
        part = {r: v for r, v in assign.items()
                if p.requests[r].kind == PICKUP or all(d in assign for d in p.requests[r].deps)}
        ev = evaluate(p, self.to_solution(part))
        ok = ev.late + ev.over + ev.dwell < 1e-9 and not (ev.deadlock or ev.overnight or ev.mixed)
        return ok, ev.km

    # ---------------------------------------------------------------- inferência
    def forward_check(self, var: str, val: Value, domains: dict, assign: dict) -> int | None:
        """Retira aos vizinhos por atribuir os valores que entram em conflito com var=val; None se um domínio ficar vazio."""
        removed = 0
        for n in self.neigh[var]:
            if n in assign:
                continue
            keep = [x for x in domains[n] if self.consistent(var, val, n, x)]
            removed += len(domains[n]) - len(keep)
            domains[n] = keep
            if not keep:
                return None
        return removed

    def ac3(self, domains: dict, assign: dict, queue=None) -> int | None:
        """Consistência de arcos sobre as restrições binárias das variáveis por atribuir."""
        if queue is None:
            queue = deque((a, b) for a in self.vars if a not in assign for b in self.neigh[a] if b not in assign)
        removed = 0
        while queue:
            a, b = queue.popleft()
            keep = [x for x in domains[a] if any(self.consistent(a, x, b, y) for y in domains[b])]
            if len(keep) < len(domains[a]):
                removed += len(domains[a]) - len(keep)
                domains[a] = keep
                if not keep:
                    return None
                queue.extend((c, a) for c in self.neigh[a] if c != b and c not in assign)
        return removed


def solve(p: Problem, inference: str = "mac", var_order: str = "mrv", value_order: str = "soft",
          optimize: bool = False, global_fc: bool = False, max_assignments: int = 200_000,
          time_limit: float = 60.0, template: tuple[str, ...] = TEMPLATE) -> CSPResult:
    """inference   : 'none' (backtracking simples) | 'fc' (forward checking) | 'mac' (FC + AC-3)
    var_order   : 'static' (ordem de entrada) | 'mrv' (menor domínio restante, desempate pelo grau)
    value_order : 'static' | 'soft' (inserção com menos km, depois least constraining value)
    optimize    : continua a pesquisa com branch & bound nos km depois da primeira solução.
    global_fc   : depois de cada atribuição retira também, a cada variável por atribuir, os valores
                  que já fazem falhar a verificação global do horário (forte, mas caro)."""
    csp = RoutingCSP(p, template)
    res = CSPResult(None, None, False, config=dict(inference=inference, var_order=var_order,
                                                   value_order=value_order, global_fc=global_fc,
                                                   optimize=optimize))
    t0 = _time.perf_counter()
    domains = {r: list(d) for r, d in csp.domains.items()}
    best = [float("inf")]

    if inference == "mac" and csp.ac3(domains, {}) is None:
        res.status, res.seconds = "inviável (AC-3 na raiz)", _time.perf_counter() - t0
        return res

    def select(assign, domains):
        free = [r for r in csp.vars if r not in assign]
        if var_order == "static":
            return free[0]
        return min(free, key=lambda r: (len(domains[r]), -sum(n not in assign for n in csp.neigh[r])))

    def order_values(var, assign, domains):
        vals = list(domains[var])
        if value_order == "static":
            return vals

        def score(val):
            # restrição suave: km acrescentados ao pôr var nessa viagem (inserção mais barata)
            members = [r for r, v in assign.items() if v == val]
            q = p.requests[var]
            if members:
                add = min(p.d(p.requests[m].loc, q.loc) for m in members) * 2
            else:
                add = 2 * p.d(p.depot, q.loc) + 50                 # abrir uma viagem nova custa mais
            # least constraining: valores retirados aos domínios dos vizinhos
            lost = sum(1 for n in csp.neigh[var] if n not in assign for x in domains[n]
                       if not csp.consistent(var, val, n, x))
            return add, lost, val
        return sorted(vals, key=score)

    def timeout():
        return res.assignments >= max_assignments or _time.perf_counter() - t0 > time_limit

    def global_filter(assign, domains) -> int | None:
        removed = 0
        for n in csp.vars:
            if n in assign:
                continue
            keep = []
            for x in domains[n]:
                assign[n] = x
                ok, km = csp.global_ok(assign)
                res.checks += 1
                if ok and km < best[0] - 1e-9:
                    keep.append(x)
                del assign[n]
            removed += len(domains[n]) - len(keep)
            domains[n] = keep
            if not keep:
                return None
        return removed

    def backtrack(assign, domains) -> bool:
        if len(assign) == len(csp.vars):
            ok, km = csp.global_ok(assign)
            res.checks += 1
            if ok and km < best[0] - 1e-9:
                best[0] = km
                res.solution, res.cost, res.feasible = csp.to_solution(assign), km, True
                res.solutions_found += 1
                res.improvements.append((res.assignments, _time.perf_counter() - t0, km))
            return not optimize                              # pára na primeira solução, exceto ao otimizar
        if timeout():
            return True
        var = select(assign, domains)
        for val in order_values(var, assign, domains):
            if inference == "none" and not all(csp.consistent(var, val, b, vb)
                                               for b, vb in assign.items() if b in csp.neigh[var]):
                continue
            res.assignments += 1
            assign[var] = val
            ok, km = csp.global_ok(assign)
            res.checks += 1
            if ok and km < best[0] - 1e-9:                   # branch & bound sobre o objetivo suave
                new = {r: list(d) for r, d in domains.items()}
                new[var] = [val]
                removed = 0
                if inference in ("fc", "mac"):
                    removed = csp.forward_check(var, val, new, assign)
                if removed is not None and inference == "mac":
                    q = deque((n, m) for n in csp.neigh[var] if n not in assign
                              for m in csp.neigh[n] if m not in assign)
                    more = csp.ac3(new, assign, q)
                    removed = None if more is None else removed + more
                if removed is not None and global_fc:
                    more = global_filter(assign, new)
                    removed = None if more is None else removed + more
                if removed is not None:
                    res.pruned += removed
                    if backtrack(assign, new):
                        return True
            del assign[var]
            res.backtracks += 1
            if timeout():
                return True
        return False

    backtrack({}, domains)
    res.seconds = _time.perf_counter() - t0
    if res.solution is None:
        res.status = "limite atingido" if timeout() else "inviável com este modelo de posições"
    else:
        res.status = "ótima para este modelo de posições" if optimize and not timeout() else (
            "melhor encontrada (limite)" if optimize else "viável")
    return res

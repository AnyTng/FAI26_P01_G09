"""CSP formulation of the routing problem.

Variables : one per request r (pickup group or order delivery).
Domain    : (vehicle, day, slot). Every day has a fixed template of trip slots, e.g.
            D P D D P D D (D = delivery trip, P = pickup trip), so a value fixes which vehicle
            serves r, in which trip, on which day. Requests sharing a value form one trip;
            the visiting order inside a trip is the shortest window-feasible permutation.
Unary     : slot kind = request kind; slot day = request day (no trip crosses midnight).
Binary    : (same trip)  capacity  w_a + w_b <= cap(v);
            (same trip)  time compatibility - a and b can follow each other within windows, and
                         (deliveries) the products of one are not ready too late for the other;
            (precedence) a delivery and one of its pickups on the same vehicle: the pickup
                         trip comes first;
            (sequence)   same vehicle, a's trip before b's trip => b must still be reachable
                         inside its window after a's trip is back at the warehouse.
Global    : trip capacity (n-ary), complete schedule (windows, warehouse precedence across
            vehicles, dwell <= 24 h) checked by the simulator on the partial assignment.
Soft      : total km (and fewer trips) - handled by the value ordering heuristic and,
            optionally, by branch & bound once a first solution is found.

Algorithms: backtracking with optional MRV + degree, value ordering (cheapest insertion /
least constraining), forward checking, MAC (AC-3) and, optionally, forward checking against
the global schedule constraint (``global_fc``).
"""
from __future__ import annotations

import itertools
import time as _time
from collections import deque
from dataclasses import dataclass, field

from .informed import _incompatible
from .model import DAY, DELIVERY, PICKUP, Problem, Solution, evaluate, simulate_trip

TEMPLATE = (DELIVERY, PICKUP, DELIVERY, DELIVERY, PICKUP, DELIVERY, DELIVERY)
Value = tuple[int, int, int]            # (vehicle id, day, slot index)


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
    improvements: list = field(default_factory=list)   # (assignments, seconds, km) per better solution

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
        # binary-constraint graph: pairs that may share a trip, delivery-pickup pairs, and pairs
        # whose trips cannot be in some order on the same vehicle
        self.neigh: dict[str, list[str]] = {r: [] for r in self.vars}
        for a in self.vars:
            for b in self.vars:
                if a != b and (self._same_group(a, b) or b in p.requests[a].deps or a in p.requests[b].deps
                               or not self.seq_ok[a, b] or not self.seq_ok[b, a]):
                    self.neigh[a].append(b)
        self._order_cache: dict[frozenset, list[str]] = {}

    # ---------------------------------------------------------------- constraints
    def _same_group(self, a: str, b: str) -> bool:
        return self.day[a] == self.day[b] and self.p.requests[a].kind == self.p.requests[b].kind

    def _time_compatible(self, a: str, b: str) -> bool:
        """Can a and b be visited one after the other (in some order) inside the windows?"""
        p, qa, qb = self.p, self.p.requests[a], self.p.requests[b]
        ab = qa.tw[0] + p.service + p.t(qa.loc, qb.loc) <= qb.tw[1]
        ba = qb.tw[0] + p.service + p.t(qb.loc, qa.loc) <= qa.tw[1]
        return ab or ba

    def _can_precede(self, a: str, b: str) -> bool:
        """Can a trip containing a come (on the same vehicle) before a trip containing b?"""
        p, qa, qb, W = self.p, self.p.requests[a], self.p.requests[b], self.p.depot
        back = qa.tw[0] + p.service + p.t(qa.loc, W) + (p.handling if qa.kind == PICKUP else 0)
        return back + (p.handling if qb.kind == DELIVERY else 0) + p.t(W, qb.loc) <= qb.tw[1] + 1e-9

    def order_key(self, val: Value) -> tuple[int, int]:
        return val[1], val[2]

    def consistent(self, a: str, va: Value, b: str, vb: Value) -> bool:
        """All binary constraints between a=va and b=vb."""
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

    def trip_order(self, members: frozenset) -> list[str]:
        """Shortest permutation of a trip that respects the windows in isolation."""
        if members in self._order_cache:
            return self._order_cache[members]
        p, W = self.p, self.p.depot
        rs = sorted(members)                    # fixed order: ties in km (reversed trips) stay stable
        if len(rs) <= 6:
            perms = list(itertools.permutations(rs))
        else:
            perms = [tuple(sorted(rs, key=lambda r: p.requests[r].tw[1]))]
        km = lambda seq: (p.d(W, p.requests[seq[0]].loc) + p.d(p.requests[seq[-1]].loc, W)
                          + sum(p.d(p.requests[x].loc, p.requests[y].loc) for x, y in zip(seq, seq[1:])))
        perms.sort(key=km)
        vid = p.vehicles[0].vid
        best = list(perms[0])
        for seq in perms:
            deps = {d: -1e9 for r in seq for d in p.requests[r].deps}
            ts = simulate_trip(p, vid, list(seq), 0.0, deps)
            if ts.late == 0:
                best = list(seq)
                break
        self._order_cache[members] = best
        return best

    def to_solution(self, assign: dict[str, Value]) -> Solution:
        trips: dict[Value, set[str]] = {}
        for r, val in assign.items():
            trips.setdefault(val, set()).add(r)
        sol: Solution = {v.vid: [] for v in self.p.vehicles}
        for val in sorted(trips, key=lambda x: (x[0],) + self.order_key(x)):
            sol[val[0]].append(self.trip_order(frozenset(trips[val])))
        return sol

    def global_ok(self, assign: dict[str, Value]) -> tuple[bool, float]:
        """Capacity of every trip + full schedule of the partial assignment (deliveries whose
        pickups are not assigned yet are left out until they are)."""
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

    # ---------------------------------------------------------------- inference
    def forward_check(self, var: str, val: Value, domains: dict, assign: dict) -> int | None:
        """Remove values of unassigned neighbours that conflict with var=val; None on wipe-out."""
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
        """Arc consistency over the binary constraints of the unassigned variables."""
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
    """inference   : 'none' (plain backtracking) | 'fc' (forward checking) | 'mac' (FC + AC-3)
    var_order   : 'static' (input order) | 'mrv' (minimum remaining values, degree tie-break)
    value_order : 'static' | 'soft' (cheapest km insertion, then least constraining value)
    optimize    : keep searching with branch & bound on km after the first solution.
    global_fc   : after each assignment also drop, from every unassigned variable, the values
                  that already make the global schedule check fail (strong but costly)."""
    csp = RoutingCSP(p, template)
    res = CSPResult(None, None, False, config=dict(inference=inference, var_order=var_order,
                                                   value_order=value_order, global_fc=global_fc,
                                                   optimize=optimize))
    t0 = _time.perf_counter()
    domains = {r: list(d) for r, d in csp.domains.items()}
    best = [float("inf")]

    if inference == "mac" and csp.ac3(domains, {}) is None:
        res.status, res.seconds = "infeasible (AC-3 at the root)", _time.perf_counter() - t0
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
            # soft constraint: km added by putting var into that trip (cheapest insertion)
            members = [r for r, v in assign.items() if v == val]
            q = p.requests[var]
            if members:
                add = min(p.d(p.requests[m].loc, q.loc) for m in members) * 2
            else:
                add = 2 * p.d(p.depot, q.loc) + 50                 # opening a new trip costs more
            # least constraining: values removed from neighbours' domains
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
            return not optimize                              # stop at the first solution unless optimizing
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
            if ok and km < best[0] - 1e-9:                   # branch & bound on the soft objective
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
        res.status = "limit reached" if timeout() else "infeasible for this slot template"
    else:
        res.status = "optimal for this template" if optimize and not timeout() else (
            "best found (limit)" if optimize else "feasible")
    return res

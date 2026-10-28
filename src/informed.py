"""Informed search (Greedy best-first and A*) over partial routing solutions.

A node is a :class:`State` (partial solution). Actions extend ONE vehicle:
  * visit an unserved request (opening a trip if the vehicle is idle at the warehouse), or
  * close the open trip (return to the warehouse).
Children violating any hard constraint (window, capacity, dwell, precedence) are discarded,
and nodes where some request can no longer be reached in time are pruned.
Path cost g = km travelled; heuristics are admissible lower bounds on the remaining km.
"""
from __future__ import annotations

import heapq
import itertools
import time as _time
from dataclasses import dataclass
from typing import Callable

from .model import DAY, DELIVERY, PICKUP, Problem, Solution, State, VState, simulate_trip

Heuristic = Callable[[Problem, State], float]


# --------------------------------------------------------------------------- successors
def successors(p: Problem, s: State):
    arrival = dict(s.arrival_w)
    unserved = [r for r in p.requests if r not in s.served]
    for i, v in enumerate(s.vehicles):
        # 1) close the open trip
        if v.kind is not None:
            ts = simulate_trip(p, v.vid, list(v.trip), v.trip_free, arrival)
            nv = VState(v.vid, p.depot, ts.end, done=v.done + (v.trip,))
            arr = s.arrival_w
            if v.kind == PICKUP:
                arr = tuple(sorted(arr + tuple((r, ts.back) for r in v.trip)))
            g = s.g - v.trip_km + ts.km
            yield State(s.vehicles[:i] + (nv,) + s.vehicles[i + 1:], s.served, arr, g)
        # 2) visit an unserved request
        for r in unserved:
            q = p.requests[r]
            if v.kind not in (None, q.kind):
                continue                                    # trips are homogeneous
            if q.kind == DELIVERY and not all(d in arrival for d in q.deps):
                continue                                    # products not yet at the warehouse
            free = v.trip_free if v.kind else v.time
            ts = simulate_trip(p, v.vid, list(v.trip) + [r], free, arrival)
            if ts.late > 0 or ts.over > 0 or ts.dwell > 0 or ts.overnight:
                continue
            km = ts.km - p.d(q.loc, p.depot)
            nv = VState(v.vid, q.loc, ts.stops[-1][2] + p.service, q.kind, v.trip + (r,), free, km, v.done)
            yield State(s.vehicles[:i] + (nv,) + s.vehicles[i + 1:], s.served | {r}, s.arrival_w, s.g - v.trip_km + km)


def dead_end(p: Problem, s: State) -> bool:
    """True if some unserved request can no longer be started inside its window by any vehicle
    (optimistic bound: direct travel, no waiting, no capacity issues). For deliveries it also
    uses the earliest moment their pickups can possibly be back at the warehouse."""
    W = p.depot
    known = dict(s.arrival_w)                                   # closed pickup trips
    at_w = dict(known)                                          # lower bound on warehouse arrival
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
    for r in pending:                                           # pickups first: deliveries depend on them
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
                # lower bound on the loading time of q's trip: its products must be ready, and the
                # trip loads just in time for whichever stop opens it - a pending delivery of the
                # same day or the first stop of an open delivery trip (not necessarily q itself)
                day = q.tw[0] // DAY
                firsts = [o.tw[0] - p.t(W, o.loc) - p.handling for o in (p.requests[x] for x in pending)
                          if o.kind == DELIVERY and o.tw[0] // DAY == day]
                firsts += [f.tw[0] - p.t(W, f.loc) - p.handling for f in
                           (p.requests[v.trip[0]] for v in s.vehicles if v.kind == DELIVERY) if f.tw[0] // DAY == day]
                if max(ready, min(firsts)) - min(closed) > p.max_dwell + 1e-9:
                    return True                                 # cannot load before the dwell limit
    return False


# --------------------------------------------------------------------------- heuristics
def h_max_detour(p: Problem, s: State) -> float:
    """h1: the farthest unserved request must still be reached by some vehicle and then
    brought back to the warehouse: max_r min_v [d(v, r) + d(r, W)] (triangle inequality)."""
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
    """(day, kind) of a request: a trip never crosses midnight and is homogeneous, so only
    requests of the same group can follow each other inside a trip."""
    q = p.requests[rid]
    return int(q.tw[0] // DAY), q.kind


def _groups(p: Problem, s: State):
    """For each (day, kind) group still active: unserved locations, open-trip vehicle locations,
    number of returns to the warehouse still needed (open trips + extra trips by weight)."""
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
    for g in sorted(nodes.keys() | opened.keys()):         # fixed order: h is a float sum
        vs = opened.get(g, [])
        spare = sum(p.vehicle(v.vid).capacity - sum(p.requests[r].weight for r in v.trip) for v in vs)
        new_trips = max(0, -(-(weight.get(g, 0.0) - spare) // cap)) if g in nodes else 0
        if g in nodes and not vs:
            pending = [r for r in p.requests if r not in s.served and _group(p, r) == g]
            new_trips = max(new_trips, 1, _clique(p, pending))
        yield g, nodes.get(g, set()), [v.loc for v in vs], len(vs) + int(new_trips)


def _incompatible(p: Problem) -> set[frozenset]:
    """Pairs of requests that can never share a trip: windows, weight, or - for deliveries - the
    earliest moment their products can be at the warehouse (a trip loads only when all are)."""
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
    """Size of a (greedy) clique of mutually incompatible requests: each needs its own trip."""
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
    """h2: relaxation on distinct (location, day, kind) nodes.
    * every unserved node needs one incoming arc, coming from the warehouse, from another node of
      the same (day, kind) group, or from a vehicle whose open trip belongs to that group;
    * every trip still open or still needed ends with one arc back to the warehouse.
    Each arc counted is distinct in any completion, so the sum of their minima is admissible."""
    W, h = p.depot, 0.0
    for g, locs, vlocs, n_ret in _groups(p, s):
        for loc in locs:
            if loc not in vlocs:                           # an open trip already standing there costs 0
                h += min([p.d(W, loc)] + [p.d(a, loc) for a in vlocs] + [p.d(b, loc) for b in locs if b != loc])
        h += n_ret * _min_return(p, locs, vlocs)
    return h


def h_mst(p: Problem, s: State) -> float:
    """h3: per (day, kind) group, removing the final return arc of every remaining trip leaves a
    forest rooted at the warehouse (open vehicles hang from it at cost 0) that spans all unserved
    locations -> its cost is >= the minimum spanning tree. Add the cheapest return arcs."""
    return sum(_mst_bounds(p, s).values())


def _mst_bounds(p: Problem, s: State) -> dict[tuple[int, str], float]:
    """The h3 bound of every active (day, kind) group."""
    W, out = p.depot, {}
    for g, locs, vlocs, n_ret in _groups(p, s):
        bound = _prim(p, {W, *vlocs}, set(locs)) + n_ret * _min_return(p, locs, vlocs)
        if locs and not vlocs:
            # 1-tree style bound: with m trips, removing the warehouse leaves at most m connected
            # pieces covering every location (>= MST minus its m-1 longest edges), and the
            # warehouse has 2m incident edges, each >= the closest location. Locations can be
            # revisited by several trips, so nothing stronger is assumed for those 2m edges.
            edges = sorted(_prim_edges(p, set(locs)), reverse=True)
            tree, w_min = sum(edges), min(p.d(W, b) for b in locs)
            lo = max(n_ret, 1)
            one_tree = min(tree - sum(edges[:m - 1]) + 2 * m * w_min for m in range(lo, max(lo, len(locs)) + 1))
            bound = max(bound, one_tree)
        out[g] = bound
    return out


class _RelaxedVRP:
    """Exact solution of a relaxation, computed once per problem and (day, kind) group.

    Relaxation: each group is served on its own by an unlimited fleet of the largest vehicle,
    every trip leaving the warehouse as early as allowed (deliveries: once the earliest possible
    return of their pickups has been unloaded and the trip loaded). Kept: windows, service and
    handling times, capacity, no overnight trips. Vehicle sharing between groups, exact pickup
    times and the dwell limit are dropped, so the optimum of the relaxation is a lower bound.

    1. Feasible trips: labelling over (subset, last request) with the earliest service time
       (exact for feasibility, since waiting is allowed) and the least km (a lower bound on
       the km of any feasible order). For deliveries one pass per "ready level" fixes the
       departure time of the trips whose latest-ready request has that level.
    2. f(U) = min-cost partition of the unserved set U into feasible trips (memoised DP).
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
                while frontier:                                   # breadth-first over subset size
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
                        continue                                  # would not be back before midnight
                    if kind == DELIVERY and max(ready[R[i]] for i in range(len(R)) if m >> i & 1) != L:
                        continue                                  # belongs to another ready level
                    trips[m] = min(trips.get(m, float("inf")), kj + p.d(Q[j].loc, W))
            low: dict[int, list[tuple[int, float]]] = {}
            for m, c in trips.items():
                low.setdefault((m & -m).bit_length() - 1, []).append((m, c))
            for lst in low.values():
                lst.sort(key=lambda x: x[1])
            self.by_low[g], self.memo[g] = low, {0: 0.0}

    def f(self, g, U: int) -> float:
        """Least km to serve the requests of mask U of group g (inf if impossible)."""
        memo = self.memo[g]
        if U in memo:
            return memo[U]
        stack, order = [U], []
        while stack:                                              # iterative DFS (no recursion limit)
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
    """h4: per (day, kind) group, the best of h3 and the exact optimum of a relaxed routing
    problem (see :class:`_RelaxedVRP`). With open trips the relaxed optimum is reduced by
    d(W, vehicle) for each one: finishing an open trip through a set S costs at least a fresh
    trip through S minus that distance (triangle inequality)."""
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
    """Edge lengths of a minimum spanning tree over ``nodes``."""
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
    """MST cost connecting ``nodes`` to the (already connected) ``roots``."""
    best = {b: min(p.d(a, b) for a in roots) for b in nodes - roots}
    total = 0.0
    while best:
        b = min(best, key=best.get)
        total += best.pop(b)
        for c in best:
            best[c] = min(best[c], p.d(b, c))
    return total


def h_zero(p: Problem, s: State) -> float:
    return 0.0


HEURISTICS = {"h1_max_detour": h_max_detour, "h2_min_arcs": h_min_arcs, "h3_mst": h_mst, "h4_relaxed": h_relaxed,
              "h0_zero": h_zero}


# --------------------------------------------------------------------------- search
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
    complete: bool              # False if stopped by the expansion limit

    def summary(self) -> dict:
        return dict(algorithm=self.algorithm, heuristic=self.heuristic,
                    cost_km=None if self.cost is None else round(self.cost, 2), expanded=self.expanded,
                    generated=self.generated, pruned=self.pruned, seconds=round(self.seconds, 3),
                    found=self.solution is not None, complete=self.complete)


def latest_loads(p: Problem) -> dict[str, float]:
    """For each pickup, the latest moment any delivery that needs it could start loading."""
    out: dict[str, float] = {}
    for q in p.requests.values():
        for d in q.deps:
            out[d] = max(out.get(d, -1e18), q.tw[1] - p.t(p.depot, q.loc) - p.handling)
    return out


def dwell_safe(p: Problem, s: State, latest: dict[str, float]) -> bool:
    """True if no pending delivery of s can ever break the dwell limit, whatever its schedule:
    then arriving earlier at the warehouse can only help, and <= dominance on arrivals is exact."""
    return all(latest.get(r, -1e18) - t <= p.max_dwell + 1e-9 for r, t in s.arrival_w)


def _dominated(store: dict, key, g: float, times: tuple) -> bool:
    """s is dominated by a stored state with the same key, lower-or-equal cost and clocks.
    Stored states are only used for this when they are dwell-safe (flag), which makes the
    pruning exact (no optimal solution is ever discarded)."""
    for g0, t0, safe in store.get(key, ()):
        if g0 <= g + 1e-9 and (all(a <= b + 1e-9 for a, b in zip(t0, times)) if safe else t0 == times):
            return True
    return False


def chronological(p: Problem, s: State, children: list[State]) -> list[State]:
    """Keep only the moves of the vehicle with the earliest clock among those that can move.
    Not used by A* (it could cut the optimum); it stops Greedy from sending one vehicle into
    the next day while the other one still has pending work."""
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
    """Generic best-first search with f = w_g * g + w_h * h.
    A* -> (1, 1); Greedy best-first -> (0, 1); Weighted A* -> (1, w>1).
    chrono=True restricts each expansion to the earliest vehicle (see :func:`chronological`).
    succ: successor function (default: stop-by-stop :func:`successors`).
    The search stops (without a proof) after max_expansions or time_limit seconds."""
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
            frontier.append(None)                           # stopped by the budget: not exhaustive
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


# --------------------------------------------------------------------------- trip-level search
@dataclass(frozen=True)
class TripOption:
    """One way of doing a trip: which requests, in which order, and how its timing depends on
    the departure time D from the warehouse (max-plus form, waiting allowed):
        back(D) = max(D + delta, c)            feasible  <=>  D <= d_max
    jit: earliest sensible departure (arrive at the first stop when its window opens)."""
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
    """All trips that can possibly be part of a solution, with dominated orders removed.

    For every (day, kind) group, orders are enumerated depth-first from the earliest possible
    departure (pickups: start of the day; deliveries: earliest moment all their products can be
    at the warehouse, plus loading). A later real departure can only make an order infeasible,
    so this is a superset of every trip of every feasible solution. Two orders of the same set
    are compared on (km, delta, c, d_max, jit): if one is at least as good on all five, it is at
    least as good for ANY departure time, any later trip and the dwell limit, so the other one is
    dropped without losing the optimum (exact reduction)."""

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
                        continue                          # infeasible now -> infeasible later too
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
        # delta: duration with no waiting; c: return time when leaving "very early".
        # Each service start is max(D + delta_k, c_k); it must be <= the window end, which needs
        # D <= tw_end - delta_k (folded into d_max below) AND c_k <= tw_end (waiting forced by
        # earlier windows) - the latter does not depend on D, so the order is simply infeasible.
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
        # d_max: latest departure that still starts every stop inside its window (backwards)
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
    """Trip-level actions: an idle vehicle performs one complete trip from the catalogue.
    The timing is computed exactly from the trip's (delta, c, d_max, jit) and the real context
    (vehicle free time, real warehouse arrivals of the pickups), with the same rules as
    :func:`simulate_trip`: windows, capacity, no overnight trip, dwell <= max_dwell."""
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
                continue                                    # no trip crosses midnight
            end = back + p.handling if o.group[1] == PICKUP else back
            nv = VState(v.vid, p.depot, end, done=v.done + (o.rids,))
            arr = s.arrival_w
            if o.group[1] == PICKUP:
                arr = tuple(sorted(arr + tuple((r, back) for r in o.rids)))
            yield State(s.vehicles[:i] + (nv,) + s.vehicles[i + 1:], s.served | set(o.rids), arr, s.g + o.km)


def astar_trips(p: Problem, h: Heuristic = None, **kw) -> SearchResult:
    """A* where each action is a whole trip (same state, cost, goal and heuristics)."""
    return best_first(p, h or h_relaxed, 1.0, 1.0, "A*-viagens", succ=trip_successors, **kw)


def greedy_trips(p: Problem, h: Heuristic = None, **kw) -> SearchResult:
    return best_first(p, h or h_relaxed, 0.0, 1.0, "Greedy-viagens", succ=trip_successors, **kw)

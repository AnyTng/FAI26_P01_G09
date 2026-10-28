"""Problem model for the local-food pickup-and-delivery routing agent.

Flow imposed by the dataset rules: producer -> collection center (warehouse) -> delivery location.

Modelling choices (documented in the main notebook):
  * Time is measured in minutes since ``start_date 00:00``.
  * A *pickup request* groups order items of the same producer whose availability
    windows intersect (interval stabbing), so one stop collects all of them.
  * A *delivery request* is one customer order (one delivery location, one window).
  * A vehicle route is a list of *trips*; every trip starts and ends at the warehouse
    and is homogeneous: only pickups or only deliveries.
  * Windows constrain the START of service (``window_semantics = service_start``).
  * A product cannot be collected before its order was placed (``order_datetime``).
  * Vehicles are back at the warehouse every night: a trip cannot cross midnight.

Sections: entities · loading · solution & schedule (cost function, feasibility checker) ·
search state · data audit and independent item-level checker · figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PICKUP, DELIVERY = "P", "D"
DAY = 24 * 60


# --------------------------------------------------------------------------- entities
@dataclass(frozen=True)
class Item:
    item_id: int
    order_id: int
    product_id: int
    producer_id: int
    weight: float
    avail: tuple[float, float]      # availability window at the producer (min)
    deliv: tuple[float, float]      # delivery window at the delivery location (min)


@dataclass(frozen=True)
class Request:
    """A stop the agent has to schedule (pickup at a producer or delivery of an order)."""
    rid: str                        # "P<n>" or "D<order_id>"
    kind: str                       # PICKUP | DELIVERY
    loc: int                        # location_id in the distance matrix
    tw: tuple[float, float]         # window for the service start (min)
    weight: float
    items: tuple[int, ...]
    deps: tuple[str, ...] = ()      # delivery: pickup requests that must reach the warehouse first


@dataclass(frozen=True)
class Vehicle:
    vid: int
    capacity: float
    refrigerated: bool
    speed: float


@dataclass
class Problem:
    name: str
    depot: int                                   # location_id of the collection center
    vehicles: list[Vehicle]
    requests: dict[str, Request]
    items: dict[int, Item]
    dist: dict[tuple[int, int], float]           # km
    time: dict[tuple[int, int], float]           # min
    service: float = 10.0                        # stop_service_min
    handling: float = 15.0                       # warehouse_handling_min (load or unload)
    max_dwell: float = 24 * 60                   # max_warehouse_dwell_h, in minutes
    start_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2026-10-30"))
    horizon_days: int = 2
    locations: pd.DataFrame | None = None
    data_dir: Path | None = None

    def __post_init__(self):
        # zero-length self arcs so the hot paths are a single dict lookup
        ids = {a for a, _ in self.dist} | {b for _, b in self.dist}
        for a in ids:
            self.dist[a, a] = self.time[a, a] = 0.0
        self._vehicles = {v.vid: v for v in self.vehicles}

    def d(self, a: int, b: int) -> float:
        return self.dist[a, b]

    def t(self, a: int, b: int) -> float:
        return self.time[a, b]

    @property
    def pickups(self) -> list[str]:
        return [r for r, q in self.requests.items() if q.kind == PICKUP]

    @property
    def deliveries(self) -> list[str]:
        return [r for r, q in self.requests.items() if q.kind == DELIVERY]

    def vehicle(self, vid: int) -> Vehicle:
        if vid not in self._vehicles:                    # fleet edited after construction
            self._vehicles = {v.vid: v for v in self.vehicles}
        return self._vehicles[vid]

    def clock(self, minutes: float) -> str:
        """Minutes since start -> 'dd HH:MM' (for printing)."""
        ts = self.start_date + pd.Timedelta(minutes=minutes)
        return ts.strftime("%d %H:%M")


# --------------------------------------------------------------------------- loading
def _group_pickups(items: list[Item], max_weight: float) -> list[list[Item]]:
    """Greedy interval stabbing: sort by window end and keep adding items while the
    windows still intersect and the weight fits. Gives the minimum number of groups
    when weight is not binding."""
    groups: list[list[Item]] = []
    for it in sorted(items, key=lambda i: (i.avail[1], i.avail[0])):
        for g in groups:
            lo = max(max(x.avail[0] for x in g), it.avail[0])
            hi = min(min(x.avail[1] for x in g), it.avail[1])
            if lo <= hi and sum(x.weight for x in g) + it.weight <= max_weight:
                g.append(it)
                break
        else:
            groups.append([it])
    return groups


def load_problem(data_dir: str | Path, grouping: str = "interval") -> Problem:
    """Read the CSVs of one dataset and build a :class:`Problem`.

    grouping = "interval" -> pickups grouped per producer by intersecting windows;
    grouping = "order"    -> one pickup per (order, producer) pair (finer, more stops).
    """
    data_dir = Path(data_dir)
    rd = lambda f: pd.read_csv(data_dir / f"{f}.csv")
    rules = rd("rules").set_index("rule")["value"]
    start = pd.Timestamp(rules["start_date"])
    mins = lambda s: (pd.to_datetime(s) - start).dt.total_seconds() / 60

    locs = rd("locations")
    loc_of = {(r.location_type, r.ref_id): r.location_id for r in locs.itertuples()}
    centers = rd("collection_centers")
    depot = loc_of["collection_center", int(centers.center_id.iloc[0])]

    veh = rd("vehicles")
    vehicles = [Vehicle(int(r.vehicle_id), float(r.capacity_kg), bool(r.refrigerated), float(r.avg_speed_kmh))
                for r in veh.itertuples()]

    products = rd("products").set_index("product_id")
    orders = rd("orders").set_index("order_id")
    oi = rd("order_items")
    for c in ("availability_start", "availability_end", "delivery_start", "delivery_end"):
        oi[c] = mins(oi[c])
    # an item cannot be collected before the customer placed the order
    ordered = mins(oi.order_id.map(orders.order_datetime))
    oi["availability_start"] = oi[["availability_start"]].assign(o=ordered).max(axis=1)
    items = {
        int(r.order_item_id): Item(int(r.order_item_id), int(r.order_id), int(r.product_id),
                                   int(products.loc[r.product_id, "producer_id"]), float(r.weight_kg),
                                   (r.availability_start, r.availability_end), (r.delivery_start, r.delivery_end))
        for r in oi.itertuples()
    }

    # pickup requests
    max_w = min(v.capacity for v in vehicles)
    by_key: dict[tuple, list[Item]] = {}
    for it in items.values():
        key = (it.producer_id,) if grouping == "interval" else (it.producer_id, it.order_id)
        by_key.setdefault(key, []).append(it)
    requests: dict[str, Request] = {}
    item_pickup: dict[int, str] = {}
    n = 0
    for key in sorted(by_key):
        for g in _group_pickups(by_key[key], max_w):
            n += 1
            rid = f"P{n}"
            tw = (max(i.avail[0] for i in g), min(i.avail[1] for i in g))
            requests[rid] = Request(rid, PICKUP, loc_of["producer", key[0]], tw,
                                    round(sum(i.weight for i in g), 3), tuple(sorted(i.item_id for i in g)))
            for i in g:
                item_pickup[i.item_id] = rid

    # delivery requests (one per order)
    for oid, row in orders.sort_index().iterrows():
        its = [i for i in items.values() if i.order_id == oid]
        tw = (max(i.deliv[0] for i in its), min(i.deliv[1] for i in its))
        deps = tuple(sorted({item_pickup[i.item_id] for i in its}, key=lambda s: int(s[1:])))
        requests[f"D{oid}"] = Request(f"D{oid}", DELIVERY, loc_of["delivery_location", int(row.delivery_location_id)],
                                      tw, round(sum(i.weight for i in its), 3),
                                      tuple(sorted(i.item_id for i in its)), deps)

    dm = rd("distance_matrix")
    dist = {(int(r.origin_location_id), int(r.destination_location_id)): float(r.distance_km) for r in dm.itertuples()}
    time = {(int(r.origin_location_id), int(r.destination_location_id)): float(r.duration_min) for r in dm.itertuples()}

    return Problem(name=data_dir.name, depot=depot, vehicles=vehicles, requests=requests, items=items,
                   dist=dist, time=time, service=float(rules["stop_service_min"]),
                   handling=float(rules["warehouse_handling_min"]),
                   max_dwell=float(rules["max_warehouse_dwell_h"]) * 60, start_date=start,
                   horizon_days=int(rules["horizon_days"]), locations=locs, data_dir=data_dir)


# --------------------------------------------------------------------------- solution & schedule
Trip = list[str]                       # sequence of request ids (warehouse implicit at both ends)
Solution = dict[int, list[Trip]]       # vehicle id -> ordered list of trips


@dataclass
class TripSchedule:
    vid: int
    kind: str
    stops: list[tuple[str, float, float]]   # (rid, arrival, service start)
    load_start: float                       # delivery trips: start of loading at the warehouse
    depart: float
    back: float                             # arrival at the warehouse
    end: float                              # vehicle free again (after unloading if pickup trip)
    km: float
    load: float                             # max load carried during the trip
    late: float = 0.0                       # minutes of window violation
    over: float = 0.0                       # kg above capacity
    dwell: float = 0.0                      # minutes above the max warehouse dwell
    mixed: bool = False
    overnight: bool = False                 # trip does not return on the day it left


def simulate_trip(p: Problem, vid: int, trip: Trip, free: float, arrival_w: dict[str, float]) -> TripSchedule:
    """Schedule one trip that leaves the warehouse when the vehicle is free at ``free``.

    ``arrival_w[pickup_rid]`` is when that pickup reached the warehouse; it is needed for
    delivery trips (precedence + dwell). Missing dependencies must be handled by the caller.
    """
    reqs = [p.requests[r] for r in trip]
    kinds = {q.kind for q in reqs}
    kind = kinds.pop() if len(kinds) == 1 else PICKUP
    cap = p.vehicle(vid).capacity
    late = over = dwell = 0.0

    if kind == DELIVERY:
        load = sum(q.weight for q in reqs)
        deps = sorted({d for q in reqs for d in q.deps})      # fixed order for the float sums
        ready = max((arrival_w[d] + p.handling for d in deps), default=free)
        # earliest moment everything is ready, but no earlier than needed to reach the first window
        load_start = max(free, ready, reqs[0].tw[0] - p.t(p.depot, reqs[0].loc) - p.handling)
        dwell = sum(max(0.0, load_start - arrival_w[d] - p.max_dwell) for d in deps)
        depart = load_start + p.handling
    else:
        load = 0.0
        first = reqs[0]
        # just-in-time: arrive at the first producer when its window opens (shorter dwell, no idle wait)
        depart = load_start = max(free, first.tw[0] - p.t(p.depot, first.loc))

    t, loc, km, peak, stops = depart, p.depot, 0.0, load, []
    for q in reqs:
        km += p.d(loc, q.loc)
        arr = t + p.t(loc, q.loc)
        start = max(arr, q.tw[0])
        late += max(0.0, start - q.tw[1])
        stops.append((q.rid, arr, start))
        load += q.weight if q.kind == PICKUP else -q.weight
        peak = max(peak, load)
        t, loc = start + p.service, q.loc
    km += p.d(loc, p.depot)
    back = t + p.t(loc, p.depot)
    over = max(0.0, peak - cap)
    end = back + p.handling if kind == PICKUP else back
    overnight = back // DAY != load_start // DAY        # vehicles are back at the warehouse every night
    return TripSchedule(vid, kind, stops, load_start, depart, back, end, km, peak, late, over, dwell,
                        len(kinds) > 0, overnight)


@dataclass
class Evaluation:
    km: float
    feasible: bool
    trips: list[TripSchedule]
    late: float = 0.0
    over: float = 0.0
    dwell: float = 0.0
    unserved: int = 0
    duplicated: int = 0
    deadlock: int = 0
    mixed: int = 0
    overnight: int = 0
    makespan: float = 0.0
    route_min: float = 0.0

    @property
    def penalized(self) -> float:
        """Cost used by local search: km plus heavy penalties for every violated constraint."""
        return (self.km + 100 * self.late + 100 * self.over + 100 * self.dwell
                + 10_000 * (self.unserved + self.duplicated + self.deadlock + self.mixed + self.overnight))

    def summary(self) -> dict:
        return dict(km=round(self.km, 2), feasible=self.feasible, trips=len(self.trips),
                    late_min=round(self.late, 1), over_kg=round(self.over, 1), dwell_excess_min=round(self.dwell, 1),
                    unserved=self.unserved, deadlock=self.deadlock, overnight=self.overnight,
                    makespan_min=round(self.makespan, 1), route_min=round(self.route_min, 1))


def evaluate(p: Problem, sol: Solution) -> Evaluation:
    """Cost function + route feasibility checker for a complete solution.

    Trips of different vehicles interact through the warehouse: a delivery trip can only
    start loading once all its pickups were unloaded. Trips are therefore simulated in a
    dependency-driven order; if no vehicle can advance, the solution is deadlocked.
    """
    seen: dict[str, int] = {}
    for trips in sol.values():
        for trip in trips:
            for r in trip:
                seen[r] = seen.get(r, 0) + 1
    unserved = sum(1 for r in p.requests if r not in seen)
    duplicated = sum(c - 1 for c in seen.values() if c > 1)

    arrival_w: dict[str, float] = {}
    free = {v.vid: 0.0 for v in p.vehicles}
    nxt = {v.vid: 0 for v in p.vehicles}
    trips_done: list[TripSchedule] = []
    queue = {vid: [t for t in sol.get(vid, []) if t] for vid in free}
    progress = True
    while progress:
        progress = False
        for vid, trips in queue.items():
            while nxt[vid] < len(trips):
                trip = trips[nxt[vid]]
                deps = {d for r in trip for d in p.requests[r].deps}
                if not deps <= arrival_w.keys():
                    break                                   # waits for another vehicle
                ts = simulate_trip(p, vid, trip, free[vid], arrival_w)
                if ts.kind == PICKUP:
                    for rid, _, _ in ts.stops:
                        arrival_w[rid] = ts.back
                free[vid] = ts.end
                trips_done.append(ts)
                nxt[vid] += 1
                progress = True
    # requests that could not be scheduled because of circular waits between vehicles
    deadlock = sum(len(trip) for v, t in queue.items() for trip in t[nxt[v]:])

    ev = Evaluation(km=sum(t.km for t in trips_done), feasible=False, trips=trips_done,
                    late=sum(t.late for t in trips_done), over=sum(t.over for t in trips_done),
                    dwell=sum(t.dwell for t in trips_done), unserved=unserved, duplicated=duplicated,
                    deadlock=deadlock, mixed=sum(t.mixed for t in trips_done),
                    overnight=sum(t.overnight for t in trips_done),
                    makespan=max((t.end for t in trips_done), default=0.0),
                    route_min=sum(t.back - t.depart for t in trips_done))
    ev.feasible = (ev.late + ev.over + ev.dwell) < 1e-6 and not (unserved or duplicated or deadlock
                                                                   or ev.mixed or ev.overnight)
    return ev


# --------------------------------------------------------------------------- search state
@dataclass(frozen=True)
class VState:
    """Situation of one vehicle inside a partial solution."""
    vid: int
    loc: int                        # current location
    time: float                     # idle: free since; on a trip: end of service at the last stop
    kind: str | None = None         # None = idle at the warehouse; else kind of the open trip
    trip: tuple[str, ...] = ()      # open trip (requests visited so far)
    trip_free: float = 0.0          # when the vehicle became free for the open trip
    trip_km: float = 0.0            # km of the open trip so far (return leg excluded)
    done: tuple[tuple[str, ...], ...] = ()   # closed trips


@dataclass(frozen=True)
class State:
    """Node of the informed search: a partial solution built trip by trip."""
    vehicles: tuple[VState, ...]
    served: frozenset[str]
    arrival_w: tuple[tuple[str, float], ...]   # pickup -> arrival time at the warehouse
    g: float                                   # km travelled so far

    @staticmethod
    def initial(p: Problem) -> "State":
        return State(tuple(VState(v.vid, p.depot, 0.0) for v in p.vehicles), frozenset(), (), 0.0)

    def is_goal(self, p: Problem) -> bool:
        return len(self.served) == len(p.requests) and all(v.kind is None for v in self.vehicles)

    def solution(self) -> Solution:
        return {v.vid: [list(t) for t in v.done] + ([list(v.trip)] if v.trip else []) for v in self.vehicles}

    def key(self) -> tuple:
        """Identity used for duplicate/dominance detection (order matters only in delivery trips,
        where adding a stop can shift the departure of the whole trip)."""
        return (self.served, tuple(r for r, _ in self.arrival_w),
                tuple((v.loc, v.kind, tuple(v.trip) if v.kind == DELIVERY else frozenset(v.trip))
                      for v in self.vehicles))

    def times(self) -> tuple[float, ...]:
        """Clocks compared for dominance: vehicles free earlier and pickups at the warehouse
        earlier. (Earlier arrivals can only hurt through the dwell limit; see informed.py.)"""
        return (tuple(v.time for v in self.vehicles) + tuple(v.trip_free for v in self.vehicles)
                + tuple(t for _, t in self.arrival_w))


def schedule_table(p: Problem, ev: Evaluation) -> pd.DataFrame:
    """Flat table with every stop of every trip (for display in notebooks)."""
    rows = []
    for k, ts in enumerate(sorted(ev.trips, key=lambda t: (t.vid, t.depart))):
        rows.append(dict(vehicle=ts.vid, trip=k, stop="W(load)" if ts.kind == DELIVERY else "W", kind=ts.kind,
                         arrival=None, start=p.clock(ts.depart), window="", weight=None))
        for rid, arr, start in ts.stops:
            q = p.requests[rid]
            rows.append(dict(vehicle=ts.vid, trip=k, stop=rid, kind=q.kind, arrival=p.clock(arr), start=p.clock(start),
                             window=f"{p.clock(q.tw[0])}-{p.clock(q.tw[1])}", weight=q.weight))
        rows.append(dict(vehicle=ts.vid, trip=k, stop="W", kind=ts.kind, arrival=p.clock(ts.back), start=None,
                         window="", weight=None))
    return pd.DataFrame(rows)


# =========================================================================== data audit & item checker
# The item-level checker does NOT use the aggregated requests: it rebuilds every order line
# straight from the CSV files and replays the explicit stop list of a solution (pickup, unload,
# load, delivery, return), so it catches mistakes in the grouping, in the simulator or in the
# search code alike. Ported (and adapted to this code base) from the group's earlier prototype.
TABLES = ["collection_centers", "customers", "delivery_locations", "distance_matrix", "localities",
          "locations", "order_items", "orders", "producers", "products", "rules", "vehicles"]
KEYS = {"collection_centers": "center_id", "customers": "customer_id", "delivery_locations": "delivery_location_id",
        "localities": "locality_id", "locations": "location_id", "order_items": "order_item_id",
        "orders": "order_id", "producers": "producer_id", "products": "product_id", "rules": "rule",
        "vehicles": "vehicle_id"}
RELATIONS = [("producers", "locality_id", "localities", "locality_id"),
             ("customers", "locality_id", "localities", "locality_id"),
             ("delivery_locations", "locality_id", "localities", "locality_id"),
             ("collection_centers", "locality_id", "localities", "locality_id"),
             ("products", "producer_id", "producers", "producer_id"),
             ("vehicles", "center_id", "collection_centers", "center_id"),
             ("orders", "customer_id", "customers", "customer_id"),
             ("orders", "delivery_location_id", "delivery_locations", "delivery_location_id"),
             ("order_items", "order_id", "orders", "order_id"),
             ("order_items", "product_id", "products", "product_id")]


def read_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(Path(data_dir) / f"{n}.csv") for n in TABLES}


# --------------------------------------------------------------------------- audit
def audit(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Integrity checks on the raw tables. Returns one row per check (ok / warning / error / info)."""
    rows = []

    def check(name, ok, detail="", severity="error"):
        rows.append(dict(check=name, status="ok" if bool(ok) else severity, detail=str(detail)))

    for name, df in tables.items():
        check(f"{name}: no missing values", not df.isna().any().any(), f"{int(df.isna().sum().sum())} empty cells")
        check(f"{name}: no duplicated rows", not df.duplicated().any())
        if name in KEYS:
            check(f"{name}: unique key", df[KEYS[name]].is_unique)
    for a, col, b, ref in RELATIONS:
        missing = set(tables[a][col]) - set(tables[b][ref])
        check(f"FK {a}.{col} -> {b}.{ref}", not missing, sorted(missing))

    loc = tables["locations"]
    check("locations: unique (type, ref)", not loc.duplicated(["location_type", "ref_id"]).any())
    for kind, table, key in [("producer", "producers", "producer_id"),
                             ("delivery_location", "delivery_locations", "delivery_location_id"),
                             ("collection_center", "collection_centers", "center_id")]:
        refs = set(loc.loc[loc.location_type == kind, "ref_id"])
        check(f"locations: covers every {kind}", refs == set(tables[table][key]))
    for t in ("locations", "localities"):
        d = tables[t]
        check(f"{t}: valid coordinates", d.latitude.between(-90, 90).all() and d.longitude.between(-180, 180).all())

    items = tables["order_items"]
    for a, b in [("availability_start", "availability_end"), ("delivery_start", "delivery_end")]:
        x, y = pd.to_datetime(items[a], errors="coerce"), pd.to_datetime(items[b], errors="coerce")
        check(f"{a} <= {b}", (x.notna() & y.notna() & (x <= y)).all())
    av, de = pd.to_datetime(items.availability_start), pd.to_datetime(items.delivery_end)
    check("availability opens before the delivery deadline", (av <= de).all())
    for t, col in [("order_items", "quantity"), ("order_items", "weight_kg"), ("products", "unit_weight_kg"),
                   ("vehicles", "capacity_kg"), ("collection_centers", "capacity_kg")]:
        check(f"{t}.{col} > 0", (tables[t][col] > 0).all())
    merged = items.merge(tables["products"], on="product_id", validate="many_to_one")
    delta = (merged.weight_kg - merged.quantity * merged.unit_weight_kg).abs()
    check("weight_kg = quantity x unit_weight_kg", (delta < 1e-6).all(), f"max deviation {delta.max():.6g} kg")
    check("every line fits in some vehicle", (items.weight_kg <= tables["vehicles"].capacity_kg.max()).all(),
          "necessary condition only", severity="warning")
    total = items.weight_kg.sum()
    check("warehouse capacity >= total demand", total <= tables["collection_centers"].capacity_kg.sum(),
          f"{total:.1f} kg", severity="warning")

    m = tables["distance_matrix"]
    ids = set(loc.location_id)
    pairs = set(zip(m.origin_location_id, m.destination_location_id))
    expected = {(a, b) for a in ids for b in ids if a != b}
    check("matrix: unique pairs", not m.duplicated(["origin_location_id", "destination_location_id"]).any())
    check("matrix: every ordered pair present", expected <= pairs, f"{len(expected - pairs)} missing")
    check("matrix: non-negative", (m[["distance_km", "duration_min"]] >= 0).all().all())
    types = loc.set_index("location_id").location_type
    check("matrix: types match locations", m.origin_location_id.map(types).eq(m.origin_type).all()
          and m.destination_location_id.map(types).eq(m.destination_type).all())
    cost = {(r.origin_location_id, r.destination_location_id): (r.distance_km, r.duration_min) for r in m.itertuples()}
    check("matrix: symmetric", all(cost.get((b, a)) == v for (a, b), v in cost.items()), severity="warning")
    zero = m[(m.distance_km == 0)]
    rows.append(dict(check="matrix: pairs at distance 0 (same coordinates)", status="info", detail=str(len(zero))))
    j = items.merge(tables["orders"][["order_id", "order_datetime"]], on="order_id")
    n = int((pd.to_datetime(j.order_datetime) > pd.to_datetime(j.availability_start)).sum())
    rows.append(dict(check="lines ordered after their pickup window opens", status="info", detail=str(n)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- item-level checker
@dataclass
class Stop:
    vid: int
    action: str                 # pickup | unload | load | delivery | return
    loc: int
    items: tuple[int, ...]
    start: float                # service start (minutes since start_date)


@dataclass
class CheckResult:
    valid: bool
    km: float
    errors: list[str] = field(default_factory=list)


def to_stops(p: Problem, ev: Evaluation) -> list[Stop]:
    """Explicit, item-level stop list of an evaluated solution."""
    out = []
    for ts in sorted(ev.trips, key=lambda t: (t.vid, t.depart)):
        items = tuple(i for rid, _, _ in ts.stops for i in p.requests[rid].items)
        if ts.kind == DELIVERY:
            out.append(Stop(ts.vid, "load", p.depot, items, ts.load_start))
        for rid, _, start in ts.stops:
            q = p.requests[rid]
            out.append(Stop(ts.vid, "pickup" if q.kind == PICKUP else "delivery", q.loc, q.items, start))
        out.append(Stop(ts.vid, "unload" if ts.kind == PICKUP else "return", p.depot,
                        items if ts.kind == PICKUP else (), ts.back))
    return out


def check_items(data_dir: str | Path, stops: list[Stop], tol: float = 1e-6) -> CheckResult:
    """Replay a stop list against the raw CSVs and report every violated rule."""
    t = read_tables(data_dir)
    rules = t["rules"].set_index("rule")["value"]
    start = pd.Timestamp(rules["start_date"])
    mins = lambda s: (pd.to_datetime(s) - start).total_seconds() / 60
    service, handling = float(rules["stop_service_min"]), float(rules["warehouse_handling_min"])
    dwell_max = float(rules["max_warehouse_dwell_h"]) * 60
    horizon = int(rules["horizon_days"]) * 24 * 60
    loc_of = {(r.location_type, r.ref_id): r.location_id for r in t["locations"].itertuples()}
    home = {r.vehicle_id: loc_of["collection_center", r.center_id] for r in t["vehicles"].itertuples()}
    cap = dict(zip(t["vehicles"].vehicle_id, t["vehicles"].capacity_kg))
    centers = {loc_of["collection_center", r.center_id]: r.capacity_kg for r in t["collection_centers"].itertuples()}
    arcs = {(r.origin_location_id, r.destination_location_id): (r.distance_km, r.duration_min)
            for r in t["distance_matrix"].itertuples()}
    prod = t["products"].set_index("product_id").producer_id
    orders = t["orders"].set_index("order_id")
    line = {}
    for r in t["order_items"].itertuples():
        o = orders.loc[r.order_id]
        line[r.order_item_id] = dict(
            pickup=loc_of["producer", prod[r.product_id]], delivery=loc_of["delivery_location", o.delivery_location_id],
            w=r.weight_kg, a=(max(mins(r.availability_start), mins(o.order_datetime)), mins(r.availability_end)),
            d=(mins(r.delivery_start), mins(r.delivery_end)))

    errors, km, events = [], 0.0, []
    count = {a: {} for a in ("pickup", "unload", "load", "delivery")}
    for vid in sorted({s.vid for s in stops}):
        if vid not in cap:
            errors.append(f"unknown vehicle {vid}")
            continue
        loc, free, onboard = home[vid], 0.0, set()
        for k, s in enumerate(x for x in stops if x.vid == vid):
            tag = f"v{vid} stop {k + 1} ({s.action} @ {s.loc})"
            if (loc, s.loc) not in arcs and loc != s.loc:
                errors.append(f"{tag}: unknown arc")
                continue
            dkm, dmin = arcs.get((loc, s.loc), (0.0, 0.0))
            km += dkm
            if s.start + tol < free + dmin:
                errors.append(f"{tag}: starts before the vehicle can arrive")
            if s.action in ("unload", "load") and s.loc not in centers:
                errors.append(f"{tag}: warehouse operation outside a collection center")
            for i in s.items:
                if i not in line:
                    errors.append(f"{tag}: unknown line {i}")
                    continue
                L = line[i]
                count[s.action][i] = count[s.action].get(i, 0) + 1 if s.action in count else 0
                if s.action == "pickup":
                    if s.loc != L["pickup"]:
                        errors.append(f"{tag}: line {i} is not at this producer")
                    if not L["a"][0] - tol <= s.start <= L["a"][1] + tol:
                        errors.append(f"{tag}: line {i} picked outside its availability window")
                if s.action == "delivery":
                    if s.loc != L["delivery"]:
                        errors.append(f"{tag}: line {i} delivered to the wrong place")
                    if not L["d"][0] - tol <= s.start <= L["d"][1] + tol:
                        errors.append(f"{tag}: line {i} delivered outside its window")
                if s.action in ("pickup", "load"):
                    if i in onboard:
                        errors.append(f"{tag}: line {i} already on board")
                    onboard.add(i)
                elif s.action in ("unload", "delivery"):
                    if i not in onboard:
                        errors.append(f"{tag}: line {i} is not on board")
                    onboard.discard(i)
                if s.action == "unload":
                    events.append((s.start + handling, 0, i, s.start, tag))   # stock available after unloading
                if s.action == "load":
                    events.append((s.start, 1, i, s.start, tag))
            load = sum(line[i]["w"] for i in onboard)
            if load > cap[vid] + tol:
                errors.append(f"{tag}: capacity exceeded ({load:.1f} kg)")
            dur = service if s.action in ("pickup", "delivery") else handling if s.action in ("unload", "load") else 0
            free, loc = s.start + dur, s.loc
            if free > horizon + tol:
                errors.append(f"{tag}: ends after the planning horizon")
        if onboard:
            errors.append(f"v{vid}: finishes with lines on board {sorted(onboard)}")
        if loc != home[vid]:
            errors.append(f"v{vid}: does not return to its collection center")

    stock, level = {}, 0.0
    for when, kind, i, arrived, tag in sorted(events):
        if kind == 0:
            stock[i] = arrived
            level += line[i]["w"]
            if level > sum(centers.values()) + tol:
                errors.append(f"{tag}: warehouse capacity exceeded")
        elif i not in stock:
            errors.append(f"{tag}: line {i} loaded before being unloaded at the warehouse")
        else:
            arrived = stock.pop(i)
            level -= line[i]["w"]
            if when - arrived > dwell_max + tol:
                errors.append(f"{tag}: line {i} stayed {(when - arrived) / 60:.1f} h in the warehouse")
    for i in line:
        for a in count:
            if count[a].get(i, 0) != 1:
                errors.append(f"line {i}: {a} done {count[a].get(i, 0)} times (expected 1)")
    return CheckResult(not errors, km, errors)


def check_solution(p: Problem, ev: Evaluation) -> CheckResult:
    return check_items(p.data_dir, to_stops(p, ev))


# =========================================================================== figures
KIND_STYLE = {"producer": ("tab:green", "^", "produtor"),
              "collection_center": ("black", "s", "centro de recolha"),
              "delivery_location": ("tab:blue", "o", "local de entrega")}
VEHICLE_COLORS = ["tab:orange", "tab:purple", "tab:red", "tab:cyan"]


def plot_locations(p: Problem, ax=None, labels: bool = True):
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    loc = p.locations
    for kind, (c, m, label) in KIND_STYLE.items():
        d = loc[loc.location_type == kind]
        ax.scatter(d.longitude, d.latitude, c=c, marker=m, s=90 if kind != "delivery_location" else 50,
                   label=label, zorder=3)
        if labels:
            for r in d.itertuples():
                ax.annotate(f"{r.location_id} {r.name}", (r.longitude, r.latitude), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_routes(p: Problem, sol: Solution, title: str = "", day: int | None = None):
    """One panel per vehicle; pickup trips dashed, delivery trips solid."""
    xy = p.locations.set_index("location_id")[["longitude", "latitude"]]
    ev = evaluate(p, sol)
    vids = sorted(sol)
    fig, axes = plt.subplots(1, len(vids), figsize=(6.5 * len(vids), 5.5), squeeze=False)
    for ax, vid in zip(axes[0], vids):
        plot_locations(p, ax, labels=False)
        trips = sorted((t for t in ev.trips if t.vid == vid), key=lambda t: t.depart)
        for k, ts in enumerate(trips):
            if day is not None and int(ts.depart // (24 * 60)) != day:
                continue
            path = [p.depot] + [p.requests[r].loc for r, _, _ in ts.stops] + [p.depot]
            pts = xy.loc[path]
            ax.plot(pts.longitude, pts.latitude, "--" if ts.kind == PICKUP else "-", lw=1.8,
                    color=plt.cm.tab10(k % 10), label=f"viagem {k + 1} ({ts.kind}) {p.clock(ts.depart)}")
        ax.set_title(f"Veículo {vid}")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(title or f"{p.name}: {ev.km:.1f} km")
    fig.tight_layout()
    return fig


def plot_gantt(p: Problem, ev: Evaluation, title: str = "", hours: tuple[int, int] = (5, 19)):
    """One panel per day. For each vehicle: trips (bars, from loading/departure to the vehicle
    being free again), service start at each stop (ticks) and the stop windows (thin lines)."""
    vids = sorted({t.vid for t in ev.trips})
    days = sorted({int(t.depart // 1440) for t in ev.trips})
    fig, axes = plt.subplots(1, len(days), figsize=(7.5 * len(days), 1.6 + 1.4 * len(vids)), squeeze=False)
    for ax, day in zip(axes[0], days):
        for ts in ev.trips:
            if int(ts.depart // 1440) != day:
                continue
            y = vids.index(ts.vid)
            color = "tab:green" if ts.kind == PICKUP else "tab:blue"
            start = ts.load_start if ts.kind == DELIVERY else ts.depart
            ax.barh(y, ts.end - start, left=start, height=0.3, color=color, alpha=0.35, edgecolor=color)
            for k, (rid, _, st) in enumerate(ts.stops):
                q = p.requests[rid]
                off = 0.22 + 0.06 * (k % 4)
                ax.plot([q.tw[0], q.tw[1]], [y + off, y + off], color=color, lw=1.2, alpha=0.8)
                ax.plot(st, y, "|", color="black", ms=12)
                ax.annotate(rid, (st, y - 0.27 - 0.1 * (k % 2)), fontsize=7, ha="center")
        ticks = [day * 1440 + h * 60 for h in range(hours[0], hours[1] + 1, 2)]
        ax.set_xticks(ticks, [p.clock(t)[3:] for t in ticks], fontsize=8)
        ax.set_xlim(day * 1440 + hours[0] * 60, day * 1440 + hours[1] * 60)
        ax.set_ylim(-0.7, len(vids) - 0.3)
        ax.set_yticks(range(len(vids)), [f"V{v}" for v in vids])
        ax.set_title(f"dia {p.clock(day * 1440)[:2]}")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title or "Horário (verde: recolhas, azul: entregas; linhas finas: janelas)")
    fig.tight_layout()
    return fig


def plot_histories(histories: dict[str, dict], title: str, key: str = "current", ax=None, cap: float | None = None):
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    for name, h in histories.items():
        ys = pd.Series(h[key])
        if cap is not None:
            ys = ys.clip(upper=cap)
        ax.plot(ys.index, ys.values, label=name, lw=1.2)
    ax.set_xlabel("iteração")
    ax.set_ylabel("custo (km, penalizado)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return ax

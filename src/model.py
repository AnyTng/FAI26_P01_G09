"""Modelo do problema para o agente de recolha e entrega de alimentos locais.

Fluxo imposto pelas regras do dataset: produtor -> centro de recolha (armazém) -> local de entrega.

Decisões de modelação (explicadas no notebook principal):
  * O tempo conta-se em minutos desde ``start_date 00:00``.
  * Um *pedido de recolha* junta as linhas do mesmo produtor cujas janelas de disponibilidade
    se intersetam (interval stabbing), para que uma só paragem as recolha todas.
  * Um *pedido de entrega* é uma encomenda (um local de entrega, uma janela).
  * A rota de um veículo é uma lista de *viagens*; cada viagem começa e acaba no armazém e é
    homogénea: só recolhas ou só entregas.
  * As janelas limitam o INÍCIO do serviço (``window_semantics = service_start``).
  * Um produto não pode ser recolhido antes de a encomenda ser feita (``order_datetime``).
  * Os veículos estão de volta ao armazém todas as noites: nenhuma viagem passa a meia-noite.

Secções: entidades · carregamento · solução e horário (função de custo, verificação de
viabilidade) · estado da pesquisa · auditoria dos dados e verificador linha a linha · figuras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PICKUP, DELIVERY = "P", "D"
DAY = 24 * 60


# --------------------------------------------------------------------------- entidades
@dataclass(frozen=True)
class Item:
    item_id: int
    order_id: int
    product_id: int
    producer_id: int
    weight: float
    avail: tuple[float, float]      # janela de disponibilidade no produtor (min)
    deliv: tuple[float, float]      # janela de entrega no local de entrega (min)


@dataclass(frozen=True)
class Request:
    """Uma paragem que o agente tem de agendar (recolha num produtor ou entrega de uma encomenda)."""
    rid: str                        # "P<n>" ou "D<order_id>"
    kind: str                       # PICKUP | DELIVERY
    loc: int                        # location_id na matriz de distâncias
    tw: tuple[float, float]         # janela para o início do serviço (min)
    weight: float
    items: tuple[int, ...]
    deps: tuple[str, ...] = ()      # entrega: recolhas que têm de chegar primeiro ao armazém


@dataclass(frozen=True)
class Vehicle:
    vid: int
    capacity: float
    refrigerated: bool
    speed: float


@dataclass
class Problem:
    name: str
    depot: int                                   # location_id do centro de recolha
    vehicles: list[Vehicle]
    requests: dict[str, Request]
    items: dict[int, Item]
    dist: dict[tuple[int, int], float]           # km
    time: dict[tuple[int, int], float]           # min
    service: float = 10.0                        # stop_service_min
    handling: float = 15.0                       # warehouse_handling_min (carregar ou descarregar)
    max_dwell: float = 24 * 60                   # max_warehouse_dwell_h, em minutos
    start_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2026-10-30"))
    horizon_days: int = 2
    locations: pd.DataFrame | None = None
    data_dir: Path | None = None
    # cache por instância das estruturas derivadas (pares incompatíveis, relaxação, catálogo de viagens)
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        # arcos de um local para si próprio com custo 0, para que os caminhos críticos sejam uma só consulta ao dicionário
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
        if vid not in self._vehicles:                    # frota alterada depois da construção
            self._vehicles = {v.vid: v for v in self.vehicles}
        return self._vehicles[vid]

    def clock(self, minutes: float) -> str:
        """Minutos desde o início -> 'dd HH:MM' (para mostrar)."""
        ts = self.start_date + pd.Timedelta(minutes=minutes)
        return ts.strftime("%d %H:%M")


# --------------------------------------------------------------------------- carregamento
def _group_pickups(items: list[Item], max_weight: float) -> list[list[Item]]:
    """Interval stabbing guloso: ordena pelo fim da janela e vai juntando linhas enquanto as
    janelas ainda se intersetam e o peso cabe. Dá o número mínimo de grupos quando o peso não
    é limitativo."""
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
    """Lê os CSV de um dataset e constrói um :class:`Problem`.

    grouping = "interval" -> recolhas agrupadas por produtor, pelas janelas que se intersetam;
    grouping = "order"    -> uma recolha por par (encomenda, produtor) (mais fino, mais paragens).
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
    # uma linha não pode ser recolhida antes de o cliente fazer a encomenda
    ordered = mins(oi.order_id.map(orders.order_datetime))
    oi["availability_start"] = oi[["availability_start"]].assign(o=ordered).max(axis=1)
    items = {
        int(r.order_item_id): Item(int(r.order_item_id), int(r.order_id), int(r.product_id),
                                   int(products.loc[r.product_id, "producer_id"]), float(r.weight_kg),
                                   (r.availability_start, r.availability_end), (r.delivery_start, r.delivery_end))
        for r in oi.itertuples()
    }

    # pedidos de recolha
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

    # pedidos de entrega (um por encomenda)
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


# --------------------------------------------------------------------------- solução e horário
Trip = list[str]                       # sequência de ids de pedidos (o armazém está implícito nas duas pontas)
Solution = dict[int, list[Trip]]       # id do veículo -> lista ordenada de viagens


@dataclass
class TripSchedule:
    vid: int
    kind: str
    stops: list[tuple[str, float, float]]   # (rid, chegada, início do serviço)
    load_start: float                       # viagens de entrega: início da carga no armazém
    depart: float
    back: float                             # chegada ao armazém
    end: float                              # veículo livre outra vez (depois de descarregar, nas viagens de recolha)
    km: float
    load: float                             # carga máxima transportada na viagem
    late: float = 0.0                       # minutos fora da janela
    over: float = 0.0                       # kg acima da capacidade
    dwell: float = 0.0                      # minutos acima da permanência máxima no armazém
    mixed: bool = False
    overnight: bool = False                 # a viagem não volta no dia em que saiu


def simulate_trip(p: Problem, vid: int, trip: Trip, free: float, arrival_w: dict[str, float]) -> TripSchedule:
    """Calcula o horário de uma viagem que sai do armazém quando o veículo fica livre, em ``free``.

    ``arrival_w[pickup_rid]`` é a hora a que essa recolha chegou ao armazém; é precisa nas
    viagens de entrega (precedência + permanência). Quem chama trata das dependências em falta.
    """
    reqs = [p.requests[r] for r in trip]
    kinds = {q.kind for q in reqs}
    kind = kinds.pop() if len(kinds) == 1 else PICKUP
    cap = p.vehicle(vid).capacity
    late = over = dwell = 0.0

    if kind == DELIVERY:
        load = sum(q.weight for q in reqs)
        deps = sorted({d for q in reqs for d in q.deps})      # ordem fixa para as somas em vírgula flutuante
        ready = max((arrival_w[d] + p.handling for d in deps), default=free)
        # primeiro momento em que está tudo pronto, mas não antes do necessário para chegar à primeira janela
        load_start = max(free, ready, reqs[0].tw[0] - p.t(p.depot, reqs[0].loc) - p.handling)
        dwell = sum(max(0.0, load_start - arrival_w[d] - p.max_dwell) for d in deps)
        depart = load_start + p.handling
    else:
        load = 0.0
        first = reqs[0]
        # just in time: chega ao primeiro produtor quando a janela abre (menos permanência, sem esperas)
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
    overnight = back // DAY != load_start // DAY        # os veículos voltam ao armazém todas as noites
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
        """Custo usado pela pesquisa local: km mais penalizações pesadas por cada restrição violada."""
        return (self.km + 100 * self.late + 100 * self.over + 100 * self.dwell
                + 10_000 * (self.unserved + self.duplicated + self.deadlock + self.mixed + self.overnight))

    def summary(self) -> dict:
        return dict(km=round(self.km, 2), feasible=self.feasible, trips=len(self.trips),
                    late_min=round(self.late, 1), over_kg=round(self.over, 1), dwell_excess_min=round(self.dwell, 1),
                    unserved=self.unserved, deadlock=self.deadlock, overnight=self.overnight,
                    makespan_min=round(self.makespan, 1), route_min=round(self.route_min, 1))


def evaluate(p: Problem, sol: Solution) -> Evaluation:
    """Função de custo + verificação de viabilidade de uma solução completa.

    As viagens de veículos diferentes interagem através do armazém: uma viagem de entrega só
    pode começar a carregar depois de descarregadas todas as suas recolhas. Por isso as viagens
    são simuladas pela ordem das dependências; se nenhum veículo puder avançar, a solução fica
    bloqueada.
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
                    break                                   # espera por outro veículo
                ts = simulate_trip(p, vid, trip, free[vid], arrival_w)
                if ts.kind == PICKUP:
                    for rid, _, _ in ts.stops:
                        arrival_w[rid] = ts.back
                free[vid] = ts.end
                trips_done.append(ts)
                nxt[vid] += 1
                progress = True
    # pedidos que não puderam ser agendados por esperas circulares entre veículos
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


# --------------------------------------------------------------------------- estado da pesquisa
@dataclass(frozen=True)
class VState:
    """Situação de um veículo numa solução parcial."""
    vid: int
    loc: int                        # local atual
    time: float                     # parado: livre desde; em viagem: fim do serviço na última paragem
    kind: str | None = None         # None = parado no armazém; senão, tipo da viagem em curso
    trip: tuple[str, ...] = ()      # viagem em curso (pedidos já visitados)
    trip_free: float = 0.0          # hora a que o veículo ficou livre para a viagem em curso
    trip_km: float = 0.0            # km da viagem em curso até agora (sem o regresso)
    done: tuple[tuple[str, ...], ...] = ()   # viagens fechadas


@dataclass(frozen=True)
class State:
    """Nó da pesquisa informada: uma solução parcial construída viagem a viagem."""
    vehicles: tuple[VState, ...]
    served: frozenset[str]
    arrival_w: tuple[tuple[str, float], ...]   # recolha -> hora de chegada ao armazém
    g: float                                   # km percorridos até agora

    @staticmethod
    def initial(p: Problem) -> "State":
        return State(tuple(VState(v.vid, p.depot, 0.0) for v in p.vehicles), frozenset(), (), 0.0)

    def is_goal(self, p: Problem) -> bool:
        return len(self.served) == len(p.requests) and all(v.kind is None for v in self.vehicles)

    def solution(self) -> Solution:
        return {v.vid: [list(t) for t in v.done] + ([list(v.trip)] if v.trip else []) for v in self.vehicles}

    def key(self) -> tuple:
        """Identidade usada para detetar duplicados e dominância (a ordem só conta nas viagens de
        entrega, onde acrescentar uma paragem pode mudar a partida da viagem toda)."""
        return (self.served, tuple(r for r, _ in self.arrival_w),
                tuple((v.loc, v.kind, tuple(v.trip) if v.kind == DELIVERY else frozenset(v.trip))
                      for v in self.vehicles))

    def times(self) -> tuple[float, ...]:
        """Relógios comparados na dominância: veículos livres mais cedo e recolhas no armazém mais
        cedo. (Chegar mais cedo só pode prejudicar através do limite de permanência; ver informed.py.)"""
        return (tuple(v.time for v in self.vehicles) + tuple(v.trip_free for v in self.vehicles)
                + tuple(t for _, t in self.arrival_w))


def schedule_table(p: Problem, ev: Evaluation) -> pd.DataFrame:
    """Tabela com todas as paragens de todas as viagens (para mostrar nos notebooks)."""
    rows = []
    for k, ts in enumerate(sorted(ev.trips, key=lambda t: (t.vid, t.depart))):
        rows.append(dict(vehicle=ts.vid, trip=k, stop="W(carga)" if ts.kind == DELIVERY else "W", kind=ts.kind,
                         arrival=None, start=p.clock(ts.depart), window="", weight=None))
        for rid, arr, start in ts.stops:
            q = p.requests[rid]
            rows.append(dict(vehicle=ts.vid, trip=k, stop=rid, kind=q.kind, arrival=p.clock(arr), start=p.clock(start),
                             window=f"{p.clock(q.tw[0])}-{p.clock(q.tw[1])}", weight=q.weight))
        rows.append(dict(vehicle=ts.vid, trip=k, stop="W", kind=ts.kind, arrival=p.clock(ts.back), start=None,
                         window="", weight=None))
    return pd.DataFrame(rows)


# =========================================================================== auditoria e verificador por linha
# O verificador por linha NÃO usa os pedidos agregados: reconstrói cada linha de encomenda
# diretamente a partir dos CSV e repete a lista explícita de paragens de uma solução (recolha,
# descarga, carga, entrega, regresso). Assim apanha erros no agrupamento, no simulador ou no
# código de pesquisa. Vem do protótipo anterior do grupo, adaptado a este código.
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


# --------------------------------------------------------------------------- auditoria
def audit(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Verificações de integridade das tabelas originais. Devolve uma linha por verificação (ok / warning / error / info)."""
    rows = []

    def check(name, ok, detail="", severity="error"):
        rows.append(dict(check=name, status="ok" if bool(ok) else severity, detail=str(detail)))

    for name, df in tables.items():
        check(f"{name}: sem valores em falta", not df.isna().any().any(), f"{int(df.isna().sum().sum())} células vazias")
        check(f"{name}: sem linhas duplicadas", not df.duplicated().any())
        if name in KEYS:
            check(f"{name}: chave única", df[KEYS[name]].is_unique)
    for a, col, b, ref in RELATIONS:
        missing = set(tables[a][col]) - set(tables[b][ref])
        check(f"FK {a}.{col} -> {b}.{ref}", not missing, sorted(missing))

    loc = tables["locations"]
    check("locations: (tipo, ref) único", not loc.duplicated(["location_type", "ref_id"]).any())
    for kind, table, key in [("producer", "producers", "producer_id"),
                             ("delivery_location", "delivery_locations", "delivery_location_id"),
                             ("collection_center", "collection_centers", "center_id")]:
        refs = set(loc.loc[loc.location_type == kind, "ref_id"])
        check(f"locations: cobre todos os {kind}", refs == set(tables[table][key]))
    for t in ("locations", "localities"):
        d = tables[t]
        check(f"{t}: coordenadas válidas", d.latitude.between(-90, 90).all() and d.longitude.between(-180, 180).all())

    items = tables["order_items"]
    for a, b in [("availability_start", "availability_end"), ("delivery_start", "delivery_end")]:
        x, y = pd.to_datetime(items[a], errors="coerce"), pd.to_datetime(items[b], errors="coerce")
        check(f"{a} <= {b}", (x.notna() & y.notna() & (x <= y)).all())
    av, de = pd.to_datetime(items.availability_start), pd.to_datetime(items.delivery_end)
    check("disponibilidade abre antes do fim da entrega", (av <= de).all())
    for t, col in [("order_items", "quantity"), ("order_items", "weight_kg"), ("products", "unit_weight_kg"),
                   ("vehicles", "capacity_kg"), ("collection_centers", "capacity_kg")]:
        check(f"{t}.{col} > 0", (tables[t][col] > 0).all())
    merged = items.merge(tables["products"], on="product_id", validate="many_to_one")
    delta = (merged.weight_kg - merged.quantity * merged.unit_weight_kg).abs()
    check("weight_kg = quantity x unit_weight_kg", (delta < 1e-6).all(), f"desvio máximo {delta.max():.6g} kg")
    check("cada linha cabe em algum veículo", (items.weight_kg <= tables["vehicles"].capacity_kg.max()).all(),
          "só condição necessária", severity="warning")
    total = items.weight_kg.sum()
    check("capacidade do armazém >= procura total", total <= tables["collection_centers"].capacity_kg.sum(),
          f"{total:.1f} kg", severity="warning")

    m = tables["distance_matrix"]
    ids = set(loc.location_id)
    pairs = set(zip(m.origin_location_id, m.destination_location_id))
    expected = {(a, b) for a in ids for b in ids if a != b}
    check("matriz: pares únicos", not m.duplicated(["origin_location_id", "destination_location_id"]).any())
    check("matriz: todos os pares ordenados presentes", expected <= pairs, f"{len(expected - pairs)} em falta")
    check("matriz: sem valores negativos", (m[["distance_km", "duration_min"]] >= 0).all().all())
    types = loc.set_index("location_id").location_type
    check("matriz: tipos coincidem com locations", m.origin_location_id.map(types).eq(m.origin_type).all()
          and m.destination_location_id.map(types).eq(m.destination_type).all())
    cost = {(r.origin_location_id, r.destination_location_id): (r.distance_km, r.duration_min) for r in m.itertuples()}
    check("matriz: simétrica", all(cost.get((b, a)) == v for (a, b), v in cost.items()), severity="warning")
    zero = m[(m.distance_km == 0)]
    rows.append(dict(check="matriz: pares à distância 0 (mesmas coordenadas)", status="info", detail=str(len(zero))))
    j = items.merge(tables["orders"][["order_id", "order_datetime"]], on="order_id")
    n = int((pd.to_datetime(j.order_datetime) > pd.to_datetime(j.availability_start)).sum())
    rows.append(dict(check="linhas encomendadas depois de abrir a janela de recolha", status="info", detail=str(n)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- verificador por linha
@dataclass
class Stop:
    vid: int
    action: str                 # pickup | unload | load | delivery | return (recolha, descarga, carga, entrega, regresso)
    loc: int
    items: tuple[int, ...]
    start: float                # início do serviço (minutos desde start_date)


@dataclass
class CheckResult:
    valid: bool
    km: float
    errors: list[str] = field(default_factory=list)


def to_stops(p: Problem, ev: Evaluation) -> list[Stop]:
    """Lista explícita de paragens, linha a linha, de uma solução avaliada."""
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
    """Repete uma lista de paragens contra os CSV originais e reporta todas as regras violadas."""
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
            errors.append(f"veículo desconhecido {vid}")
            continue
        loc, free, onboard = home[vid], 0.0, set()
        for k, s in enumerate(x for x in stops if x.vid == vid):
            tag = f"v{vid} paragem {k + 1} ({s.action} @ {s.loc})"
            if (loc, s.loc) not in arcs and loc != s.loc:
                errors.append(f"{tag}: arco desconhecido")
                continue
            dkm, dmin = arcs.get((loc, s.loc), (0.0, 0.0))
            km += dkm
            if s.start + tol < free + dmin:
                errors.append(f"{tag}: começa antes de o veículo poder chegar")
            if s.action in ("unload", "load") and s.loc not in centers:
                errors.append(f"{tag}: operação de armazém fora de um centro de recolha")
            for i in s.items:
                if i not in line:
                    errors.append(f"{tag}: linha desconhecida {i}")
                    continue
                L = line[i]
                count[s.action][i] = count[s.action].get(i, 0) + 1 if s.action in count else 0
                if s.action == "pickup":
                    if s.loc != L["pickup"]:
                        errors.append(f"{tag}: a linha {i} não é deste produtor")
                    if not L["a"][0] - tol <= s.start <= L["a"][1] + tol:
                        errors.append(f"{tag}: linha {i} recolhida fora da janela de disponibilidade")
                if s.action == "delivery":
                    if s.loc != L["delivery"]:
                        errors.append(f"{tag}: linha {i} entregue no local errado")
                    if not L["d"][0] - tol <= s.start <= L["d"][1] + tol:
                        errors.append(f"{tag}: linha {i} entregue fora da janela")
                if s.action in ("pickup", "load"):
                    if i in onboard:
                        errors.append(f"{tag}: a linha {i} já está a bordo")
                    onboard.add(i)
                elif s.action in ("unload", "delivery"):
                    if i not in onboard:
                        errors.append(f"{tag}: a linha {i} não está a bordo")
                    onboard.discard(i)
                if s.action == "unload":
                    events.append((s.start + handling, 0, i, s.start, tag))   # stock disponível depois de descarregar
                if s.action == "load":
                    events.append((s.start, 1, i, s.start, tag))
            load = sum(line[i]["w"] for i in onboard)
            if load > cap[vid] + tol:
                errors.append(f"{tag}: capacidade excedida ({load:.1f} kg)")
            dur = service if s.action in ("pickup", "delivery") else handling if s.action in ("unload", "load") else 0
            free, loc = s.start + dur, s.loc
            if free > horizon + tol:
                errors.append(f"{tag}: acaba depois do horizonte de planeamento")
        if onboard:
            errors.append(f"v{vid}: acaba com linhas a bordo {sorted(onboard)}")
        if loc != home[vid]:
            errors.append(f"v{vid}: não volta ao seu centro de recolha")

    stock, level = {}, 0.0
    for when, kind, i, arrived, tag in sorted(events):
        if kind == 0:
            stock[i] = arrived
            level += line[i]["w"]
            if level > sum(centers.values()) + tol:
                errors.append(f"{tag}: capacidade do armazém excedida")
        elif i not in stock:
            errors.append(f"{tag}: linha {i} carregada antes de ser descarregada no armazém")
        else:
            arrived = stock.pop(i)
            level -= line[i]["w"]
            if when - arrived > dwell_max + tol:
                errors.append(f"{tag}: a linha {i} ficou {(when - arrived) / 60:.1f} h no armazém")
    for i in line:
        for a in count:
            if count[a].get(i, 0) != 1:
                errors.append(f"linha {i}: {a} feito {count[a].get(i, 0)} vezes (esperado 1)")
    return CheckResult(not errors, km, errors)


def check_solution(p: Problem, ev: Evaluation) -> CheckResult:
    return check_items(p.data_dir, to_stops(p, ev))


# =========================================================================== figuras
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
    """Um painel por veículo; viagens de recolha a tracejado, de entrega a cheio."""
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
    """Um painel por dia. Para cada veículo: viagens (barras, da carga ou partida até o veículo
    ficar livre outra vez), início do serviço em cada paragem (traços) e janelas (linhas finas)."""
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

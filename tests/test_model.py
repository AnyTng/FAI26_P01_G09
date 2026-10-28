"""Casos sintéticos para o modelo, o simulador e o verificador linha a linha.

O fixture escreve numa pasta temporária um dataset com 1 produtor, 1 centro e 2 destinos, para
testar o mesmo carregador de CSV que é usado nos datasets reais.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model import audit, check_solution, evaluate, load_problem, read_tables

ROOT = Path(__file__).resolve().parents[1]


def write_fixture(folder: Path, capacity: float = 50.0, deliv_end: str = "2026-10-30 14:00:00",
                  order_time: str = "2026-10-29 20:00:00", deliv_start: str = "2026-10-30 09:00:00") -> Path:
    """Locais: 1 produtor, 2 centro, 3 e 4 entrega. Cada arco = 10 km / 10 min."""
    tables = {
        "localities": pd.DataFrame(dict(locality_id=[1], name=["X"], municipality=["Braga"], district=["Braga"],
                                        latitude=[41.5], longitude=[-8.4])),
        "producers": pd.DataFrame(dict(producer_id=[1], name=["P"], locality_id=[1], address=["a"], phone=["1"],
                                       email=["p@x.pt"])),
        "products": pd.DataFrame(dict(product_id=[1, 2], producer_id=[1, 1], name=["Kiwi", "Pera"],
                                      category=["fruit", "fruit"], unit=["kg", "kg"], unit_weight_kg=[1.0, 1.0])),
        "collection_centers": pd.DataFrame(dict(center_id=[1], name=["C"], locality_id=[1], address=["c"],
                                                capacity_kg=[1000.0])),
        "vehicles": pd.DataFrame(dict(vehicle_id=[1], plate=["AA"], center_id=[1], capacity_kg=[capacity],
                                      refrigerated=[0], avg_speed_kmh=[45.0])),
        "customers": pd.DataFrame(dict(customer_id=[1, 2], name=["A", "B"], locality_id=[1, 1], phone=["1", "2"],
                                       email=["a@x", "b@x"])),
        "delivery_locations": pd.DataFrame(dict(delivery_location_id=[1, 2], name=["L1", "L2"], locality_id=[1, 1],
                                                address=["x", "y"], location_type=["home", "store"])),
        "orders": pd.DataFrame(dict(order_id=[1, 2], customer_id=[1, 2], delivery_location_id=[1, 2],
                                    order_datetime=[order_time, "2026-10-29 20:00:00"], status=["confirmed"] * 2)),
        "order_items": pd.DataFrame(dict(
            order_item_id=[1, 2], order_id=[1, 2], product_id=[1, 2], quantity=[30.0, 30.0], weight_kg=[30.0, 30.0],
            availability_start=["2026-10-30 08:00:00"] * 2, availability_end=["2026-10-30 10:00:00"] * 2,
            delivery_start=[deliv_start] * 2, delivery_end=[deliv_end] * 2)),
        "locations": pd.DataFrame(dict(location_id=[1, 2, 3, 4],
                                       location_type=["producer", "collection_center", "delivery_location",
                                                      "delivery_location"],
                                       ref_id=[1, 1, 1, 2], name=["P", "C", "L1", "L2"], municipality=["Braga"] * 4,
                                       latitude=[41.5] * 4, longitude=[-8.4] * 4)),
        "rules": pd.DataFrame(dict(rule=["start_date", "horizon_days", "flow", "stop_service_min",
                                         "warehouse_handling_min", "max_warehouse_dwell_h", "window_semantics",
                                         "avg_speed_kmh", "route_factor", "generator_seed"],
                                   value=["2026-10-30", "2", "producer>warehouse>delivery_location", "10", "15", "24",
                                          "service_start", "45.0", "1.3", "1"],
                                   description=[""] * 10)),
    }
    types = dict(zip(tables["locations"].location_id, tables["locations"].location_type))
    tables["distance_matrix"] = pd.DataFrame(
        [dict(origin_location_id=a, origin_type=types[a], destination_location_id=b, destination_type=types[b],
              distance_km=10.0, duration_min=10.0) for a in types for b in types if a != b])
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)
    return folder


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def problem(self, **kw):
        return load_problem(write_fixture(self.dir, **kw))


class ModelTests(Fixture):
    def test_grouping_and_requests(self):
        p = self.problem(capacity=60)
        self.assertEqual(sorted(p.requests), ["D1", "D2", "P1"])           # as duas linhas recolhidas numa só paragem
        self.assertEqual(p.requests["P1"].weight, 60.0)
        self.assertEqual(p.requests["D1"].deps, ("P1",))

    def test_feasible_solution_and_cost(self):
        p = self.problem(capacity=60)
        ev = evaluate(p, {1: [["P1"], ["D1", "D2"]]})
        self.assertTrue(ev.feasible, ev.summary())
        self.assertAlmostEqual(ev.km, 20 + 30)                           # W-P-W + W-L1-L2-W
        self.assertTrue(check_solution(p, ev).valid)

    def test_capacity_violation(self):
        p = self.problem(capacity=50)                                      # o grupo de recolha de 60 kg é dividido
        self.assertEqual(len(p.pickups), 2)
        ev = evaluate(p, {1: [["P1", "P2"], ["D1", "D2"]]})
        self.assertGreater(ev.over, 0)
        self.assertFalse(check_solution(p, ev).valid)

    def test_delivery_needs_warehouse_first(self):
        p = self.problem(capacity=60)
        ev = evaluate(p, {1: [["D1", "D2"], ["P1"]]})                     # viagem de entrega antes da recolha
        self.assertFalse(ev.feasible)
        self.assertGreater(ev.deadlock, 0)

    def test_late_delivery(self):
        p = self.problem(capacity=60, deliv_start="2026-10-30 08:30:00", deliv_end="2026-10-30 08:40:00")
        ev = evaluate(p, {1: [["P1"], ["D1", "D2"]]})
        self.assertGreater(ev.late, 0)
        self.assertFalse(check_solution(p, ev).valid)

    def test_order_datetime_delays_pickup(self):
        p = self.problem(capacity=60, order_time="2026-10-30 09:00:00")
        self.assertEqual(p.items[1].avail[0], 9 * 60)                     # não se recolhe antes de a encomenda existir

    def test_unserved_and_duplicates_are_penalised(self):
        p = self.problem(capacity=60)
        self.assertEqual(evaluate(p, {1: [["P1"], ["D1"]]}).unserved, 1)
        self.assertEqual(evaluate(p, {1: [["P1"], ["D1", "D2", "D1"]]}).duplicated, 1)

    def test_cross_vehicle_transfer(self):
        p = self.problem(capacity=60)
        p.vehicles.append(type(p.vehicles[0])(2, 60.0, False, 45.0))
        ev = evaluate(p, {2: [["D1", "D2"]], 1: [["P1"]]})                # o veículo 2 espera pelo veículo 1
        self.assertTrue(ev.feasible, ev.summary())


class DatasetTests(unittest.TestCase):
    def test_audit_has_no_errors(self):
        for ds in ("tiny", "small"):
            a = audit(read_tables(ROOT / "data" / ds))
            self.assertFalse((a.status == "error").any(), a[a.status == "error"])

    def test_every_line_belongs_to_one_pickup_and_one_delivery(self):
        for ds in ("tiny", "small"):
            p = load_problem(ROOT / "data" / ds)
            for kind in ("P", "D"):
                lines = [i for q in p.requests.values() if q.kind == kind for i in q.items]
                self.assertEqual(sorted(lines), sorted(p.items))


if __name__ == "__main__":
    unittest.main()

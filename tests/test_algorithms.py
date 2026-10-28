"""Algorithms on the synthetic fixture and on tiny (fast settings only)."""
import unittest
from pathlib import Path

from src.csp import solve
from src.informed import HEURISTICS, State, astar, astar_trips, greedy, h_mst, h_zero, trip_successors
from src.local_search import hill_climbing, simulated_annealing
from src.model import (DAY, DELIVERY, PICKUP, Problem, Request, Vehicle, check_solution, evaluate, load_problem,
                       simulate_trip)
from tests.test_model import Fixture

ROOT = Path(__file__).resolve().parents[1]


class SearchOnFixture(Fixture):
    def test_astar_equals_uniform_cost(self):
        p = self.problem(capacity=60)
        ucs = astar(p, h_zero)
        self.assertTrue(ucs.complete)
        for name, h in HEURISTICS.items():
            r = astar(p, h)
            self.assertAlmostEqual(r.cost, ucs.cost, msg=name)
            self.assertLessEqual(h(p, State.initial(p)), ucs.cost + 1e-9, msg=name)   # admissible at the root
            self.assertTrue(check_solution(p, evaluate(p, r.solution)).valid)

    def test_infeasible_instance_is_exhausted(self):
        p = self.problem(capacity=60, deliv_start="2026-10-30 08:30:00", deliv_end="2026-10-30 08:40:00")
        r = astar(p)
        self.assertIsNone(r.solution)
        self.assertTrue(r.complete)

    def test_expansion_limit_is_not_infeasibility(self):
        r = astar(self.problem(capacity=60), max_expansions=0)
        self.assertIsNone(r.solution)
        self.assertFalse(r.complete)


class DwellPruning(unittest.TestCase):
    def test_dead_end_keeps_trip_opened_by_earlier_delivery(self):
        """D2 alone would load too late for the 24 h limit, but a trip opened by D1 (earlier window)
        loads in time and waits at D2. The dwell pruning must not discard that solution."""
        H = 60
        T = {(0, 1): 30, (0, 2): 10, (0, 3): 20, (1, 2): 35, (1, 3): 45, (2, 3): 10}
        time = {k: float(v) for (a, b), v in T.items() for k in ((a, b), (b, a))}
        dist = {k: 0.75 * v for k, v in time.items()}
        reqs = {"P1": Request("P1", PICKUP, 1, (8 * H, 8 * H + 10), 10.0, (1, 2)),
                "D1": Request("D1", DELIVERY, 2, (DAY + 7 * H, DAY + 12 * H), 5.0, (1,), ("P1",)),
                "D2": Request("D2", DELIVERY, 3, (DAY + 9.5 * H, DAY + 12 * H), 5.0, (2,), ("P1",))}
        p = Problem("dwell", 0, [Vehicle(1, 100.0, False, 45.0)], reqs, {}, dist, time)
        manual = evaluate(p, {1: [["P1"], ["D1", "D2"]]})
        self.assertTrue(manual.feasible, manual.summary())
        for r in (astar(p, h_zero), astar_trips(p, h_mst)):
            self.assertIsNotNone(r.solution, r.summary())
            self.assertAlmostEqual(r.cost, manual.km)


class PerProblemCaches(unittest.TestCase):
    def test_heuristics_do_not_leak_between_problems(self):
        """Derived structures are cached on the Problem itself, so a new instance (even one that
        reuses the memory address of a freed one) never sees another instance's data."""
        expected = {ds: h_mst(load_problem(ROOT / "data" / ds), State.initial(load_problem(ROOT / "data" / ds)))
                    for ds in ("tiny", "small")}
        for ds in ("small", "tiny") * 3:
            p = load_problem(ROOT / "data" / ds)
            self.assertAlmostEqual(h_mst(p, State.initial(p)), expected[ds], msg=ds)
            del p


class TripLevelSearch(Fixture):
    def test_trip_astar_equals_stop_astar(self):
        p = self.problem(capacity=60)
        self.assertAlmostEqual(astar_trips(p).cost, astar(p, h_zero).cost)


class TripLevelOnTiny(unittest.TestCase):
    def test_trip_timing_matches_simulator(self):
        """Every child of the trip-level search must match simulate_trip exactly."""
        p = load_problem(ROOT / "data" / "tiny")
        frontier, seen = [State.initial(p)], 0
        while frontier and seen < 300:
            s = frontier.pop()
            for c in trip_successors(p, s):
                seen += 1
                i = next(k for k, (a, b) in enumerate(zip(s.vehicles, c.vehicles)) if a is not b)
                trip = list(c.vehicles[i].done[-1])
                ts = simulate_trip(p, c.vehicles[i].vid, trip, s.vehicles[i].time, dict(s.arrival_w))
                self.assertFalse(ts.late or ts.over or ts.dwell or ts.overnight, trip)
                self.assertAlmostEqual(ts.end, c.vehicles[i].time, places=6)
                self.assertAlmostEqual(ts.km, c.g - s.g, places=6)
                frontier.append(c)

    def test_trip_astar_finds_tiny_optimum(self):
        p = load_problem(ROOT / "data" / "tiny")
        r = astar_trips(p)
        self.assertTrue(r.complete)
        self.assertAlmostEqual(r.cost, 533.09, places=2)
        self.assertTrue(check_solution(p, evaluate(p, r.solution)).valid)


class OnTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load_problem(ROOT / "data" / "tiny")
        cls.g = greedy(cls.p)

    def test_greedy_is_valid(self):
        self.assertIsNotNone(self.g.solution)
        ev = evaluate(self.p, self.g.solution)
        self.assertTrue(ev.feasible)
        self.assertTrue(check_solution(self.p, ev).valid)

    def test_heuristics_never_exceed_known_solution(self):
        s, cost = State.initial(self.p), self.g.cost
        for name, h in HEURISTICS.items():
            self.assertLessEqual(h(self.p, s), cost + 1e-9, msg=name)

    def test_hill_climbing_never_gets_worse(self):
        for mode in ("steepest", "first"):
            r = hill_climbing(self.p, self.g.solution, mode)
            self.assertTrue(r.feasible)
            self.assertLessEqual(r.cost, self.g.cost + 1e-9)
            self.assertTrue(all(a >= b for a, b in zip(r.history, r.history[1:])))

    def test_simulated_annealing_returns_best_feasible(self):
        r = simulated_annealing(self.p, self.g.solution, iters=2000)
        self.assertTrue(r.feasible)
        self.assertLessEqual(r.cost, self.g.cost + 1e-9)

    def test_csp_modes_find_valid_assignments(self):
        for inference in ("none", "fc", "mac"):
            r = solve(self.p, inference=inference)
            self.assertTrue(r.feasible, r.summary())
            self.assertTrue(check_solution(self.p, evaluate(self.p, r.solution)).valid)


if __name__ == "__main__":
    unittest.main()

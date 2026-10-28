"""Algorithms on the synthetic fixture and on tiny (fast settings only)."""
import unittest
from pathlib import Path

from src.csp import solve
from src.informed import HEURISTICS, State, astar, astar_trips, greedy, h_zero, trip_successors
from src.local_search import hill_climbing, simulated_annealing
from src.model import check_solution, evaluate, load_problem, simulate_trip
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

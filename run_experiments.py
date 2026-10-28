"""Reproduz todas as experiências do projeto e grava os resultados em reports/.

    python run_experiments.py            # execução completa (o A* exato no tiny demora vários minutos)
    python run_experiments.py --quick    # orçamentos pequenos, para uma verificação rápida

Os notebooks leem estes ficheiros, para não ser preciso repetir as pesquisas caras sempre que
se abre um notebook. Cada solução volta a ser verificada com o verificador linha a linha.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd

from src.csp import solve
from src.informed import HEURISTICS, astar, astar_trips, catalog, greedy, weighted_astar
from src.local_search import hill_climbing, simulated_annealing
from src.model import audit, check_solution, evaluate, load_problem, read_tables, schedule_table

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
H = {k: v for k, v in HEURISTICS.items() if k != "h0_zero"}


def checked(p, sol) -> dict:
    """Simulador + verificação independente, linha a linha, de uma solução."""
    ev = evaluate(p, sol)
    chk = check_solution(p, ev) if ev.feasible else None
    return dict(feasible=ev.feasible, item_check=bool(chk and chk.valid), km_check=None if chk is None else round(chk.km, 2))


def save_solution(name: str, p, sol) -> None:
    (REPORTS / f"solution_{name}.json").write_text(json.dumps({str(v): t for v, t in sol.items()}, indent=1))
    schedule_table(p, evaluate(p, sol)).to_csv(REPORTS / f"schedule_{name}.csv", index=False)


def informed(p, ds: str, quick: bool) -> tuple[list[dict], dict]:
    rows, sols = [], {}
    limit_astar = 20_000 if quick else (1_000_000 if ds == "tiny" else 150_000)
    budget = dict(max_expansions=limit_astar, time_limit=60 if quick else 600)
    runs = [("Greedy", hn, lambda h: greedy(p, h, max_expansions=100_000)) for hn in H]
    runs += [("A*", hn, lambda h: astar(p, h, **budget)) for hn in H]
    runs += [(f"WA*(w={w})", "h3_mst", (lambda w: lambda h: weighted_astar(p, h, w, **budget))(w))
             for w in (1.2, 1.5, 2.0)]
    # A* por viagens (uma ação = uma viagem completa do catálogo)
    runs += [("A*-viagens", hn, lambda h: astar_trips(p, h, max_expansions=200_000, time_limit=60 if quick else 600))
             for hn in ("h3_mst", "h4_relaxed")]
    for alg, hn, run in runs:
        r = run(H[hn])
        row = dict(dataset=ds, **r.summary(), heuristic_name=hn)
        if r.solution:
            row.update(checked(p, r.solution))
            sols[f"{alg}_{hn}"] = r.solution
        rows.append(row)
        print(ds, row, flush=True)
    return rows, sols


def local(p, ds: str, start, quick: bool) -> tuple[list[dict], dict, dict]:
    rows, hist, sols = [], {}, {}
    sa_iters = 3_000 if quick else 30_000
    runs = [("HC-steepest", lambda s: hill_climbing(p, start, "steepest", seed=s), [0]),
            ("HC-first", lambda s: hill_climbing(p, start, "first", seed=s), range(5)),
            ("SA", lambda s: simulated_annealing(p, start, t0=30, alpha=0.9998, iters=sa_iters, seed=s), range(5))]
    for name, run, seeds in runs:
        for seed in seeds:
            r = run(seed)
            row = {"dataset": ds, "seed": seed, **r.summary(), **checked(p, r.solution)}
            rows.append(row)
            hist[f"{name}_{seed}"] = dict(current=r.history, best=r.best_history)
            sols[f"{name}_{seed}"] = (r.cost, r.solution)
            print(ds, row, flush=True)
    return rows, hist, sols


def csp(p, ds: str, quick: bool) -> tuple[list[dict], dict]:
    rows, sols = [], {}
    budget = dict(max_assignments=5_000 if quick else 20_000, time_limit=20 if quick else 90)
    for inference in ("none", "fc", "mac"):
        for var_order in ("static", "mrv"):
            for value_order in ("static", "soft"):
                for gfc in (False, True):
                    r = solve(p, inference=inference, var_order=var_order, value_order=value_order,
                              global_fc=gfc, **budget)
                    row = dict(dataset=ds, **r.summary())
                    if r.solution:
                        row.update(checked(p, r.solution))
                        sols[f"csp_{inference}_{var_order}_{value_order}_{int(gfc)}"] = r.solution
                    rows.append(row)
                    print(ds, row, flush=True)
    r = solve(p, global_fc=True, optimize=True, max_assignments=2_000 if quick else 20_000,
              time_limit=60 if quick else 600)
    row = dict(dataset=ds, **r.summary())
    if r.solution:
        row.update(checked(p, r.solution))
        sols["csp_branch_and_bound"] = r.solution
    rows.append(row)
    print(ds, row, flush=True)
    return rows, sols


def main(quick: bool) -> None:
    REPORTS.mkdir(exist_ok=True)
    tables = {"informed": [], "local": [], "csp": [], "audit": []}
    histories, best, catalogs = {}, {}, {}

    def flush():                                   # grava o que já existe (se houver um crash, fica guardado)
        for name, rows in tables.items():
            if rows and name != "audit":
                pd.DataFrame(rows).to_csv(REPORTS / f"{name}.csv", index=False)
    for ds in ("tiny", "small"):
        p = load_problem(ROOT / "data" / ds)
        a = audit(read_tables(ROOT / "data" / ds))
        tables["audit"].append(a.assign(dataset=ds))

        rows, sols = informed(p, ds, quick)
        tables["informed"] += rows
        flush()
        start_name = "Greedy_h3_mst"
        start = sols[start_name]
        rows, hist, lsols = local(p, ds, start, quick)
        tables["local"] += rows
        histories[ds] = hist
        flush()
        crows, csols = csp(p, ds, quick)
        tables["csp"] += crows
        flush()

        # melhor solução de cada família + melhor de todas
        candidates = {**{k: (evaluate(p, s).km, s) for k, s in sols.items()}, **lsols,
                      **{k: (evaluate(p, s).km, s) for k, s in csols.items()}}
        for k, (km, s) in candidates.items():
            if evaluate(p, s).feasible and (ds not in best or km < best[ds][0]):
                best[ds] = (km, s, k)
        save_solution(f"{ds}_greedy", p, start)
        for k in ("A*-viagens_h4_relaxed", "A*_h4_relaxed", "A*_h3_mst"):
            if k in sols:
                save_solution(f"{ds}_astar", p, sols[k])
                break
        cat = catalog(p)
        catalogs[ds] = dict(raw_orders=cat.raw_orders, options=len(cat.options))
        save_solution(f"{ds}_best", p, best[ds][1])

    pd.concat(tables.pop("audit")).to_csv(REPORTS / "audit.csv", index=False)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(REPORTS / f"{name}.csv", index=False)
    (REPORTS / "local_histories.json").write_text(json.dumps(histories))
    (REPORTS / "best.json").write_text(json.dumps({ds: dict(km=round(km, 2), source=k) for ds, (km, _, k) in best.items()},
                                                  indent=1))
    (REPORTS / "environment.json").write_text(json.dumps(dict(
        python=sys.version, platform=platform.platform(), pandas=pd.__version__, quick=quick,
        trip_catalog=catalogs), indent=1))
    print("concluído:", {ds: (round(km, 2), k) for ds, (km, _, k) in best.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="orçamentos de pesquisa pequenos")
    main(ap.parse_args().quick)

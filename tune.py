"""
Tune – förbättra CL Predictor och bevisa att det blev bättre.

1. Delar upp förra säsongens matcher:
     TRÄNING = ligafasen   (modellen får justeras på dessa)
     TEST    = slutspelet  (modellen får ALDRIG se dessa när den justeras)
2. Söker de bästa värdena för modellens parametrar på träningsdatan.
3. Jämför version 1 (nuvarande värden) med version 2 (nya värden) på testdatan.

Kör (lägg filen i samma mapp som cl_predict.py och backtest.py):
  python tune.py                 # säsong 2025/26
  python tune.py --season 2025
  python tune.py --demo
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import asdict
from datetime import date, timedelta

from rich.align import Align
from rich.columns import Columns
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import backtest as bt
import cl_predict as cp

console = cp.console
PARAMS_FILE = "model_params.json"
RESULTS_MD = "tune_results.md"

# Sökrutnätet – vilka värden som provas
GRID = {
    "elo_to_goals": [0.0016, 0.0020, 0.0024, 0.0028, 0.0032, 0.0036, 0.0040, 0.0044, 0.0048],
    "home_adv": [0, 25, 50, 75, 100],
    "base_goals": [1.30, 1.40, 1.50, 1.60],
    "k_factor": [10, 20, 30, 40],
}

PARAM_NAMES = {
    "elo_to_goals": ("ELO_TO_GOALS", "hur mycket lagstyrkan påverkar xG"),
    "home_adv": ("HOME_ADV_ELO", "hemmafördel i Elo-poäng"),
    "base_goals": ("BASE_GOALS", "snittmål per lag"),
    "k_factor": ("K_FACTOR", "hur snabbt Elo reagerar på resultat"),
}


# ----------------------------------------------------------------------------
# Data (sparas lokalt så att API:t bara anropas en gång)
# ----------------------------------------------------------------------------
def load_matches(args):
    if args.demo:
        return bt.demo_matches(240), "demo"
    cache = f"matches_{args.season}.json"
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
        matches = [bt.Played(date.fromisoformat(m["day"]), m["home"], m["away"], m["hg"],
                             m["ag"], m["neutral"], m["stage"]) for m in raw["matches"]]
        return matches, raw["label"]
    key = os.getenv("FOOTBALL_DATA_KEY") or cp.API_KEY
    with console.status("[bold]Hämtar matcher…", spinner="dots12"):
        matches, label = bt.fetch_matches(key, args.season)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"label": str(label), "matches": [
            {**asdict(m), "day": m.day.isoformat()} for m in matches]}, f, ensure_ascii=False)
    return matches, label


def split(matches):
    """Ligafas = träning, slutspel = test. Saknas slutspel: de sista 25 % blir test."""
    test = [m.stage != "LEAGUE_STAGE" for m in matches]
    if any(test) and not all(test):
        return test, "ligafasen", "slutspelet"
    cut = int(len(matches) * 0.75)
    return [i >= cut for i in range(len(matches))], "första 75 %", "sista 25 %"


# ----------------------------------------------------------------------------
# Modell med valfria parametrar
# ----------------------------------------------------------------------------
def pmf(lam: float, n: int = 10):
    out = [math.exp(-lam)]
    for k in range(1, n + 1):
        out.append(out[-1] * lam / k)
    return out


def probs(eh, ea, neutral, p):
    diff = eh + (0 if neutral else p["home_adv"]) - ea
    lh = p["base_goals"] * math.exp(p["elo_to_goals"] * diff / 2)
    la = p["base_goals"] * math.exp(-p["elo_to_goals"] * diff / 2)
    h, a = pmf(lh), pmf(la)
    cdf_a, c = [], 0.0
    for v in a:
        c += v
        cdf_a.append(c)
    ph = sum(h[i] * cdf_a[i - 1] for i in range(1, len(h)))
    pd = sum(h[i] * a[i] for i in range(len(h)))
    pa = max(1.0 - ph - pd, 1e-9)
    t = ph + pd + pa
    return ph / t, pd / t, pa / t


def evaluate(matches, test_mask, start, p, mode, collect=False):
    """Kör igenom säsongen i ordning. Returnerar mått för träning och test."""
    R = dict(start)
    acc = {"train": [0.0, 0.0, 0.0, 0.0, 0], "test": [0.0, 0.0, 0.0, 0.0, 0]}
    pairs = []
    for m, is_test in zip(matches, test_mask):
        if mode == "clubelo":
            table = bt.elo_on(m.day - timedelta(days=1)) or {}
            eh, ea = bt.team_elo(m.home, table), bt.team_elo(m.away, table)
        else:
            eh, ea = R[m.home], R[m.away]
        pr = probs(eh, ea, m.neutral, p)
        o = m.outcome
        s = acc["test" if is_test else "train"]
        s[0] += bt.rps(pr, o)
        s[1] += bt.brier(pr, o)
        s[2] += bt.logloss(pr, o)
        s[3] += 1.0 if max(range(3), key=lambda k: pr[k]) == o else 0.0
        s[4] += 1
        if collect and is_test:
            pairs += [(pr[k], 1.0 if k == o else 0.0) for k in range(3)]
        if mode == "rullande":
            exp_h = 1 / (1 + 10 ** (-(R[m.home] + (0 if m.neutral else p["home_adv"])
                                      - R[m.away]) / 400))
            res = (1.0, 0.5, 0.0)[o]
            gd = abs(m.hg - m.ag)
            mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            d = p["k_factor"] * mult * (res - exp_h)
            R[m.home] += d
            R[m.away] -= d
    out = {}
    for name, (r, b, l, a, n) in acc.items():
        out[name] = {"rps": r / n, "brier": b / n, "log": l / n, "acc": a / n, "n": n}
    return (out, pairs) if collect else out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Förbättra CL Predictor")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    console.print(Panel(Align.center(Text("STEG 3 – förbättra modellen\nträna på en del, testa på resten",
                                          style="bold white", justify="center")),
                        border_style="magenta"))

    matches, label = load_matches(args)
    test_mask, train_name, test_name = split(matches)
    n_test = sum(test_mask)
    console.print(f"Säsong [bold]{label}[/] • träning: [bold]{train_name}[/] "
                  f"({len(matches) - n_test} matcher) • test: [bold]{test_name}[/] ({n_test} matcher)")

    # Elo-läge och startvärden
    teams = {m.home for m in matches} | {m.away for m in matches}
    mode = "rullande"
    if not args.demo and bt.elo_on(matches[0].day - timedelta(days=1)):
        mode = "clubelo"
    start_table = cp.FALLBACK_ELO
    if mode == "rullande" and not args.demo and os.path.exists("elo_cache.csv"):
        import csv
        with open("elo_cache.csv", encoding="utf-8") as f:
            start_table = {row["Club"]: float(row["Elo"]) for row in csv.DictReader(f)}
    start = {t: bt.team_elo(t, start_table) for t in teams}
    console.print(f"[dim]Elo-läge: {mode}[/]\n")

    grid = dict(GRID)
    if mode == "clubelo":
        grid["k_factor"] = [20]          # K används inte när clubelo ger Elo per dag

    v1 = {"elo_to_goals": cp.ELO_TO_GOALS, "home_adv": cp.HOME_ADV_ELO,
          "base_goals": cp.BASE_GOALS, "k_factor": bt.K_FACTOR}

    # Sök bästa kombinationen – BARA på träningsdatan
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    best_p, best_rps = None, float("inf")
    with Progress(SpinnerColumn("dots12"), TextColumn("[bold]Provar kombinationer"),
                  BarColumn(bar_width=36), TextColumn("{task.completed}/{task.total}"),
                  TextColumn("[green]{task.fields[best]}"), console=console,
                  transient=True) as prog:
        task = prog.add_task("", total=len(combos), best="")
        for p in combos:
            r = evaluate(matches, test_mask, start, p, mode)["train"]["rps"]
            if r < best_rps:
                best_rps, best_p = r, p
            prog.update(task, advance=1, best=f"bästa RPS hittills: {best_rps:.4f}")

    # Jämför v1 och v2 – nu på testdatan
    res1, pairs1 = evaluate(matches, test_mask, start, v1, mode, collect=True)
    res2, pairs2 = evaluate(matches, test_mask, start, best_p, mode, collect=True)

    console.print(Rule("[bold cyan]Nya parametrar[/]"))
    pt = Table(header_style="bold magenta", border_style="blue")
    pt.add_column("Parameter")
    pt.add_column("Betyder", style="dim")
    pt.add_column("Version 1", justify="right")
    pt.add_column("Version 2", justify="right", style="bold green")
    for k, (const, desc) in PARAM_NAMES.items():
        if mode == "clubelo" and k == "k_factor":
            continue
        pt.add_row(const, desc, f"{v1[k]:g}", f"{best_p[k]:g}")
    console.print(pt)

    console.print(Rule("[bold cyan]Version 1 mot version 2[/]"))
    mt = Table(header_style="bold magenta", border_style="blue")
    mt.add_column("")
    mt.add_column("RPS ↓", justify="right")
    mt.add_column("Brier ↓", justify="right")
    mt.add_column("Log loss ↓", justify="right")
    mt.add_column("Träff 1X2 ↑", justify="right")
    for part, title in (("train", f"Träning ({train_name})"), ("test", f"TEST ({test_name})")):
        for ver, res in (("v1", res1), ("v2", res2)):
            s = res[part]
            style = "bold green" if ver == "v2" and part == "test" else ""
            mt.add_row(Text(f"{title} – {ver}", style=style), f"{s['rps']:.4f}",
                       f"{s['brier']:.4f}", f"{s['log']:.4f}", f"{s['acc']:.1%}")
        if part == "train":
            mt.add_section()
    console.print(mt)

    t1, t2 = res1["test"]["rps"], res2["test"]["rps"]
    gain = (1 - t2 / t1) * 100
    tr_gain = (1 - res2["train"]["rps"] / res1["train"]["rps"]) * 100
    if gain > 1:
        msg = (f"✅ Version 2 har [bold]{gain:.1f} % lägre fel[/] på testmatcherna, "
               f"som modellen aldrig fick se när den justerades.")
        color = "green"
    elif gain > -1:
        msg = (f"➖ Ingen tydlig skillnad på testmatcherna ({gain:+.1f} %). "
               f"Förbättringen på träningen ({tr_gain:+.1f} %) håller inte på ny data.")
        color = "yellow"
    else:
        msg = (f"⚠️ Version 2 är [bold]sämre[/] på testmatcherna ({gain:+.1f} %) trots "
               f"{tr_gain:+.1f} % på träningen. Det kallas överanpassning – behåll version 1.")
        color = "red"
    if n_test < 60:
        msg += f"\n[dim]Obs: testet har bara {n_test} matcher, så siffran är osäker.[/]"
    console.print(Panel(msg, border_style=color))

    console.print(Columns([
        _titled(bt.calibration_table(pairs1), "Kalibrering på test – version 1"),
        _titled(bt.calibration_table(pairs2), "Kalibrering på test – version 2"),
    ]))

    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump({"version": 2, "season_trained": str(label), "params": best_p,
                   "test_rps_v1": round(t1, 4), "test_rps_v2": round(t2, 4)}, f, indent=2)

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(f"# Resultat – säsong {label}\n\n")
        f.write(f"Träning: {train_name} ({len(matches) - n_test} matcher). "
                f"Test: {test_name} ({n_test} matcher).\n\n")
        f.write("| | RPS | Brier | Log loss | Träff 1X2 |\n|---|---|---|---|---|\n")
        for ver, res in (("Version 1", res1), ("Version 2", res2)):
            s = res["test"]
            f.write(f"| {ver} (test) | {s['rps']:.4f} | {s['brier']:.4f} | "
                    f"{s['log']:.4f} | {s['acc']:.1%} |\n")
        f.write(f"\nVersion 2 har {abs(gain):.1f} % {'lägre' if gain >= 0 else 'högre'} "
                f"RPS-fel än version 1 på testdatan.\n")

    console.print(f"\n[bold]Klistra in i cl_predict.py[/] om version 2 vann (ersätt raderna):")
    console.print(f"[cyan]BASE_GOALS = {best_p['base_goals']:g}\n"
                  f"ELO_TO_GOALS = {best_p['elo_to_goals']:g}\n"
                  f"HOME_ADV_ELO = {best_p['home_adv']:g}[/]")
    if mode == "rullande":
        console.print(f"[cyan]K_FACTOR = {best_p['k_factor']:g}[/]  [dim](i backtest.py)[/]")
    console.print(f"\n[dim]Sparat: {PARAMS_FILE} och {RESULTS_MD} (till README:n i steg 2).[/]")


def _titled(table: Table, title: str) -> Table:
    table.title = title
    return table


if __name__ == "__main__":
    main()
"""
CL Season – vägen till finalen i Madrid 2027.

Simulerar resten av Champions League-säsongen tusentals gånger:
  ligafas (36 lag, 8 matcher var) -> playoff -> åttondel -> kvart -> semi -> final i Madrid

Redan spelade matcher hämtas från football-data.org och räknas som de slutade.
Allt som inte är spelat simuleras med samma modell som cl_predict.py (Elo -> xG -> Poisson).

Kör (lägg filen i samma mapp som cl_predict.py):
  python cl_season.py               # riktig säsong
  python cl_season.py --demo        # påhittad säsong
  python cl_season.py --sims 20000  # fler simuleringar = stabilare procent
  python cl_season.py --snabb       # hoppa över animationerna
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import requests
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import cl_predict as cp   # återanvänder nyckel, Elo-hämtning och modellparametrar

console = cp.console

FINAL_VENUE = "Estadio Metropolitano, Madrid"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@dataclass
class Fixture:
    home: object
    away: object
    matchday: int
    hg: int | None = None
    ag: int | None = None

    @property
    def played(self) -> bool:
        return self.hg is not None


def fetch_season(api_key: str):
    """Hämtar alla ligafasmatcher för innevarande säsong från football-data."""
    r = requests.get("https://api.football-data.org/v4/competitions/CL/matches",
                     headers={"X-Auth-Token": api_key}, timeout=20)
    r.raise_for_status()
    teams, fixtures = {}, []
    for m in r.json().get("matches", []):
        if m.get("stage") != "LEAGUE_STAGE":
            continue
        h, a = m["homeTeam"], m["awayTeam"]
        if not h.get("id") or not a.get("id"):
            continue
        teams[h["id"]] = h.get("shortName") or h["name"]
        teams[a["id"]] = a.get("shortName") or a["name"]
        ft = (m.get("score") or {}).get("fullTime") or {}
        done = m.get("status") == "FINISHED" and ft.get("home") is not None
        fixtures.append(Fixture(h["id"], a["id"], m.get("matchday") or 0,
                                ft["home"] if done else None, ft["away"] if done else None))
    return teams, fixtures


def demo_season():
    """36 påhittade lag-scheman: varje lag 8 motståndare, 4 hemma och 4 borta.
    Omgång 1–4 är 'redan spelade'."""
    names = list(cp.FALLBACK_ELO.keys())[:36]
    random.shuffle(names)
    teams = {n: n for n in names}
    fixtures = []
    for i in range(36):
        for k in range(1, 5):
            a, b = names[i], names[(i + k) % 36]
            home, away = (a, b) if k % 2 else (b, a)
            fixtures.append(Fixture(home, away, 2 * k - 1 + (i % 2)))
    ratings = {n: cp.FALLBACK_ELO[n] for n in names}
    for f in fixtures:
        if f.matchday <= 4:
            f.hg, f.ag = play(f.home, f.away, ratings)
    XG_CACHE.clear()
    return teams, fixtures, ratings


def build_ratings(teams: dict, use_demo: bool):
    """Elo per lag. Om clubelo är nere justeras reserv-Elo efter årets CL-resultat."""
    if use_demo:
        return None, True
    elo = cp.fetch_elo(datetime.now(cp.TZ).date())
    ratings = {}
    for tid, name in teams.items():
        val = cp.lookup_elo(name, elo)
        if val is None:
            console.print(f"[dim]{name} saknas i Elo-listan – sätter 1650.[/]")
            val = 1650.0
        ratings[tid] = float(val)
    return ratings, elo is cp.FALLBACK_ELO


def apply_results_to_elo(ratings: dict, fixtures: list[Fixture], k: float = 10.0):
    """Uppdaterar Elo med säsongens spelade matcher (vinst mot starkt lag = mer)."""
    for f in sorted((f for f in fixtures if f.played), key=lambda f: f.matchday):
        exp_h = 1 / (1 + 10 ** (-(ratings[f.home] + cp.HOME_ADV_ELO - ratings[f.away]) / 400))
        res = 1.0 if f.hg > f.ag else 0.5 if f.hg == f.ag else 0.0
        gd = abs(f.hg - f.ag)
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        d = k * mult * (res - exp_h)
        ratings[f.home] += d
        ratings[f.away] -= d


# ----------------------------------------------------------------------------
# Matchmotor
# ----------------------------------------------------------------------------
XG_CACHE: dict = {}


def xg_pair(h, a, R, neutral=False):
    key = (h, a, neutral)
    if key not in XG_CACHE:
        diff = R[h] + (0 if neutral else cp.HOME_ADV_ELO) - R[a]
        XG_CACHE[key] = (cp.BASE_GOALS * math.exp(cp.ELO_TO_GOALS * diff / 2),
                         cp.BASE_GOALS * math.exp(-cp.ELO_TO_GOALS * diff / 2))
    return XG_CACHE[key]


def pois(lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p < limit:
            return k
        k += 1


def play(h, a, R, neutral=False):
    lh, la = xg_pair(h, a, R, neutral)
    return pois(lh), pois(la)


def tie(hi, lo, R, log=None):
    """Dubbelmöte. Lägre seedat lag har hemma först, högre seedat avgör hemma."""
    g1_lo, g1_hi = play(lo, hi, R)
    g2_hi, g2_lo = play(hi, lo, R)
    tot_hi, tot_lo = g1_hi + g2_hi, g1_lo + g2_lo
    extra = ""
    if tot_hi == tot_lo:
        lh, ll = xg_pair(hi, lo, R)
        e_hi, e_lo = pois(lh / 3), pois(ll / 3)          # förlängning = 30 min
        tot_hi, tot_lo = tot_hi + e_hi, tot_lo + e_lo
        if e_hi or e_lo:
            extra = "e.fl."
        if tot_hi == tot_lo:
            extra = "straffar"
            winner = hi if random.random() < 0.5 else lo
        else:
            winner = hi if tot_hi > tot_lo else lo
    else:
        winner = hi if tot_hi > tot_lo else lo
    if log is not None:
        log.append((hi, lo, (g1_hi, g1_lo), (g2_hi, g2_lo), (tot_hi, tot_lo), winner, extra))
    return winner


def final(a, b, R, log=None):
    """Finalen: en match på neutral plan."""
    ga, gb = play(a, b, R, neutral=True)
    extra = ""
    if ga == gb:
        la, lb = xg_pair(a, b, R, True)
        ea, eb = pois(la / 3), pois(lb / 3)
        ga, gb = ga + ea, gb + eb
        if ea or eb:
            extra = "e.fl."
        if ga == gb:
            extra = "straffar"
            winner = random.choice((a, b))
        else:
            winner = a if ga > gb else b
    else:
        winner = a if ga > gb else b
    if log is not None:
        log["Final"] = (a, b, ga, gb, winner, extra)
    return winner


# ----------------------------------------------------------------------------
# Säsong
# ----------------------------------------------------------------------------
def league_table(teams, fixtures, R):
    pts, gf, ga = defaultdict(int), defaultdict(int), defaultdict(int)
    for f in fixtures:
        hg, ag = (f.hg, f.ag) if f.played else play(f.home, f.away, R)
        gf[f.home] += hg; ga[f.home] += ag
        gf[f.away] += ag; ga[f.away] += hg
        if hg > ag:
            pts[f.home] += 3
        elif hg < ag:
            pts[f.away] += 3
        else:
            pts[f.home] += 1; pts[f.away] += 1
    order = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t], random.random()),
                   reverse=True)
    return order, pts, gf, ga


def knockout(order, R, log=None):
    """Nya formatet: 1–8 direkt till åttondel, 9–24 playoff, 25–36 ute.
    Seedningen följer UEFA:s upplägg (förenklad lottning)."""
    seed = lambda n: order[n - 1]
    rank = {t: i for i, t in enumerate(order)}

    # Playoff: 9/10 mot 23/24, 11/12 mot 21/22, 13/14 mot 19/20, 15/16 mot 17/18
    po, lg = {}, ([] if log is not None else None)
    for hi_pair, lo_pair in (((9, 10), (23, 24)), ((11, 12), (21, 22)),
                             ((13, 14), (19, 20)), ((15, 16), (17, 18))):
        lows = list(lo_pair)
        random.shuffle(lows)
        for h, l in zip(hi_pair, lows):
            po[h] = tie(seed(h), seed(l), R, lg)
    if log is not None:
        log["Playoff"] = lg

    # Åttondel: 1/2 möter vinnare från 15–18, 3/4 från 13/14–19/20 osv.
    # 1 och 2 hamnar på olika halvor och kan bara mötas i finalen.
    top, bottom = [], []
    for pair, opp_from in (((1, 2), (15, 16)), ((7, 8), (9, 10)),
                           ((3, 4), (13, 14)), ((5, 6), (11, 12))):
        p = list(pair); random.shuffle(p)
        o = [po[opp_from[0]], po[opp_from[1]]]; random.shuffle(o)
        top.append((seed(p[0]), o[0]))
        bottom.append((seed(p[1]), o[1]))
    r16 = top + bottom

    def run(pairs, name):
        lg = [] if log is not None else None
        winners = []
        for x, y in pairs:
            hi, lo = (x, y) if rank[x] < rank[y] else (y, x)
            winners.append(tie(hi, lo, R, lg))
        if log is not None:
            log[name] = lg
        return winners

    qf = run(r16, "Åttondelsfinal")
    sf = run(list(zip(qf[::2], qf[1::2])), "Kvartsfinal")
    fin = run(list(zip(sf[::2], sf[1::2])), "Semifinal")
    champ = final(fin[0], fin[1], R, log)
    return {"r16": [t for p in r16 for t in p], "qf": qf, "sf": sf, "final": fin, "champ": champ}


def monte_carlo(teams, fixtures, R, n: int):
    stats = {t: defaultdict(float) for t in teams}
    with Progress(SpinnerColumn("dots12"), TextColumn("[bold]Simulerar säsonger"),
                  BarColumn(bar_width=40), TextColumn("{task.completed:,}/{task.total:,}"),
                  TimeRemainingColumn(), console=console, transient=True) as prog:
        task = prog.add_task("", total=n)
        for i in range(n):
            order, pts, _, _ = league_table(teams, fixtures, R)
            ko = knockout(order, R)
            for pos, t in enumerate(order, 1):
                s = stats[t]
                s["pos"] += pos
                s["pts"] += pts[t]
                if pos <= 8:
                    s["top8"] += 1
                elif pos <= 24:
                    s["po"] += 1
            for key in ("r16", "qf", "sf", "final"):
                for t in ko[key]:
                    stats[t][key] += 1
            stats[ko["champ"]]["win"] += 1
            if i % 200 == 0:
                prog.update(task, completed=i)
        prog.update(task, completed=n)
    for s in stats.values():
        for k in s:
            s[k] /= n
    return stats


# ----------------------------------------------------------------------------
# Visning
# ----------------------------------------------------------------------------
TROPHY = "\n".join([
    "   ___________",
    "  '._==_==_=_.'",
    "  .-\\:      /-.",
    " | (|:.     |) |",
    "  '-|:.     |-'",
    "    \\::.    /",
    "     '::. .'",
    "       ) (",
    "     _.' '._",
    "    `-------`",
])


def heat(p: float) -> Text:
    if p < 0.005:
        return Text("–", style="dim")
    style = ("bold white on dark_green" if p >= 0.75 else "bold green" if p >= 0.45
             else "green" if p >= 0.2 else "yellow" if p >= 0.05 else "dim")
    return Text(f"{p:.0%}" if p >= 0.01 else "<1%", style=style)


def intro(skip: bool):
    if skip:
        return
    target = "MADRID 2027"
    for i in range(32):
        line = " " * i + "⚽" + " " * (32 - i) + "🏆 " + target
        console.print(f"[bold green]{line}[/]", end="\r")
        time.sleep(0.03)
    console.print(" " * 60, end="\r")
    console.print(Panel(Align.center(Text("VÄGEN TILL MADRID 2027\nUEFA Champions League • hela säsongen",
                                          style="bold white", justify="center")),
                        border_style="magenta"))


def current_standings(teams, fixtures):
    pts, played = defaultdict(int), defaultdict(int)
    for f in fixtures:
        if not f.played:
            continue
        played[f.home] += 1; played[f.away] += 1
        if f.hg > f.ag:
            pts[f.home] += 3
        elif f.hg < f.ag:
            pts[f.away] += 3
        else:
            pts[f.home] += 1; pts[f.away] += 1
    return pts, played


def prob_table(teams, stats, R, fixtures, n):
    pts_now, played = current_standings(teams, fixtures)
    rows = sorted(teams, key=lambda t: (-stats[t]["win"], -stats[t]["final"], -stats[t]["sf"],
                                        -stats[t]["qf"], -stats[t]["r16"], stats[t]["pos"]))
    tbl = Table(title=f"Sannolikheter efter {n:,} simulerade säsonger", title_style="bold",
                header_style="bold magenta", border_style="blue", expand=True)
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("Lag", no_wrap=True, overflow="ellipsis", ratio=2)
    tbl.add_column("Elo", justify="right")
    tbl.add_column("Nu", justify="right")
    tbl.add_column("Snitt-\nplats", justify="right")
    tbl.add_column("Snitt-\npoäng", justify="right")
    tbl.add_column("Topp 8", justify="right")
    tbl.add_column("Play-\noff", justify="right")
    tbl.add_column("Åttondel", justify="right")
    tbl.add_column("Kvart", justify="right")
    tbl.add_column("Semi", justify="right")
    tbl.add_column("Final", justify="right")
    tbl.add_column("Vinst", justify="right")
    tbl.add_column("", width=10, no_wrap=True)
    top_win = max(stats[t]["win"] for t in teams) or 1
    for i, t in enumerate(rows, 1):
        s = stats[t]
        bar = cp.bar(s["win"] / top_win, 10, "yellow")
        tbl.add_row(str(i), teams[t], f"{R[t]:.0f}", f"{pts_now[t]}p/{played[t]}",
                    f"{s['pos']:.1f}", f"{s['pts']:.1f}",
                    heat(s["top8"]), heat(s["po"]), heat(s["r16"]), heat(s["qf"]),
                    heat(s["sf"]), heat(s["final"]), heat(s["win"]), bar)
    return tbl


def show_featured(teams, fixtures, R, skip: bool):
    """Spelar upp EN möjlig säsong steg för steg."""
    log = {}
    order, pts, gf, ga = league_table(teams, fixtures, R)
    knockout(order, R, log)
    name = lambda t: teams[t]
    pause = 0 if skip else 1

    console.print(Rule("[bold magenta]🎬 En möjlig säsong – spelas upp[/]"))

    # Ligatabell, avslöjas rad för rad
    def table_upto(k):
        t = Table(title="Ligafas – sluttabell", title_style="bold", box=None,
                  header_style="bold")
        for col in ("#", "Lag", "P", "MS", ""):
            t.add_column(col, justify="right" if col in ("#", "P", "MS") else "left",
                         no_wrap=True)
        for i, team in enumerate(order[:k], 1):
            zone = ("[bold green]→ Åttondel[/]" if i <= 8 else
                    "[yellow]→ Playoff[/]" if i <= 24 else "[red]Ute[/]")
            t.add_row(str(i), name(team), str(pts[team]), f"{gf[team] - ga[team]:+d}", zone)
        return t

    with Live(table_upto(0), console=console, refresh_per_second=30) as live:
        for k in range(1, len(order) + 1):
            live.update(table_upto(k))
            time.sleep(0.04 * pause)
    console.print()

    # Slutspelsrundor
    for rnd in ("Playoff", "Åttondelsfinal", "Kvartsfinal", "Semifinal"):
        console.print(Rule(f"[bold cyan]{rnd}[/]"))
        for hi, lo, l1, l2, tot, w, extra in log[rnd]:
            if not skip:
                with Live(Text(f"  {name(hi)} – {name(lo)}  ⏳", style="dim"),
                          console=console, transient=True):
                    time.sleep(0.35)
            line = Text()
            line.append(f"  {name(hi):>18} ", style="bold" if w == hi else "dim")
            line.append(f" {tot[0]}–{tot[1]} ", style="bold white on blue")
            line.append(f" {name(lo):<18}", style="bold" if w == lo else "dim")
            line.append(f"  (1:a {l1[0]}–{l1[1]}, 2:a {l2[0]}–{l2[1]}"
                        + (f", {extra}" if extra else "") + ")", style="dim")
            line.append(f"  → {name(w)}", style="bold green")
            console.print(line)
        time.sleep(0.4 * pause)

    # Finalen
    a, b, g_a, g_b, w, extra = log["Final"]
    console.print(Rule(f"[bold yellow]🏟  FINAL – {FINAL_VENUE}[/]"))
    if not skip:
        with Live(console=console, refresh_per_second=20, transient=True) as live:
            for minute in range(0, 91, 3):
                live.update(Align.center(Text(f"{name(a)}  vs  {name(b)}   {minute}'",
                                              style="bold yellow")))
                time.sleep(0.07)
    res = f"{name(a)}  {g_a} – {g_b}  {name(b)}" + (f"   ({extra})" if extra else "")
    console.print(Align.center(Text(res, style="bold white on blue")))

    champ = Text(f"🏆  {name(w).upper()} VINNER CHAMPIONS LEAGUE 2027  🏆",
                 style="bold black on yellow", justify="center")
    if skip:
        console.print(Panel(Group(Align.center(Text(TROPHY, style="yellow")), champ),
                            border_style="yellow"))
        return
    with Live(console=console, refresh_per_second=10) as live:
        for i in range(14):
            style = "yellow" if i % 2 else "bold white"
            live.update(Panel(Group(Align.center(Text(TROPHY, style=style)), champ),
                              border_style="yellow" if i % 2 else "magenta"))
            time.sleep(0.15)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Simulera hela CL-säsongen till finalen i Madrid")
    ap.add_argument("--demo", action="store_true", help="påhittad säsong")
    ap.add_argument("--sims", type=int, default=10000, help="antal simuleringar (standard 10 000)")
    ap.add_argument("--seed", type=int, help="slumpfrö för att kunna upprepa en körning")
    ap.add_argument("--snabb", action="store_true", help="hoppa över animationerna")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    key = cp.os.getenv("FOOTBALL_DATA_KEY") or cp.API_KEY
    demo = args.demo or not key

    intro(args.snabb)

    if demo:
        console.print("[yellow]Demo-läge – påhittade lag och resultat.[/]")
        teams, fixtures, R = demo_season()
        fallback = False
    else:
        with console.status("[bold]Hämtar säsongens matcher…", spinner="dots12"):
            teams, fixtures = fetch_season(key)
        if not fixtures:
            console.print("[red]Hittade inga ligafasmatcher i API:t.[/]")
            return
        with console.status("[bold]Hämtar Elo-ratingar…", spinner="dots12"):
            R, fallback = build_ratings(teams, False)

    n_played = sum(f.played for f in fixtures)
    console.print(f"[bold]{len(teams)} lag[/] • [bold]{n_played}[/] av {len(fixtures)} "
                  f"ligamatcher spelade • resten simuleras")

    if fallback:
        apply_results_to_elo(R, fixtures)
        console.print("[dim]Reserv-Elo har justerats efter årets spelade CL-matcher.[/]")
    XG_CACHE.clear()

    stats = monte_carlo(teams, fixtures, R, args.sims)
    console.print(prob_table(teams, stats, R, fixtures, args.sims))

    fav = max(teams, key=lambda t: stats[t]["win"])
    console.print(Panel(f"Störst chans att vinna i Madrid: [bold yellow]{teams[fav]}[/] "
                        f"({stats[fav]['win']:.0%})", border_style="yellow"))

    show_featured(teams, fixtures, R, args.snabb)

    console.print("\n[dim]Modell: Elo → xG → Poisson, Monte Carlo. Lottningen till slutspelet "
                  "är förenklad. Underhållning, inte spelråd.[/]")


if __name__ == "__main__":
    main()
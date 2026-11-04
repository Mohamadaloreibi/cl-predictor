"""
Backtest – hur bra är CL Predictor egentligen?

Låter modellen tippa matcher som redan är spelade, UTAN att veta resultatet,
och jämför sedan med facit. Mäter:
  * RPS (Ranked Probability Score) – standardmåttet för fotbollsprognoser, lägre = bättre
  * Brier score                     – felet i sannolikheterna, lägre = bättre
  * Log loss                        – straffar självsäkra misstag hårt, lägre = bättre
  * Träffsäkerhet                   – hur ofta mest troliga utfallet (1/X/2) inträffade
och jämför mot två baslinjer som modellen måste slå.

Inget fusk: modellen får bara veta lagens styrka FÖRE varje match.
  * Om clubelo.com svarar används Elo från dagen före varje match.
  * Annars "rullande Elo": startvärden som uppdateras efter varje spelad match.

Kör (lägg filen i samma mapp som cl_predict.py):
  python backtest.py                 # förra säsongen (om API:t tillåter), annars innevarande
  python backtest.py --season 2025   # välj säsong (startår)
  python backtest.py --demo          # påhittade matcher, för att testa programmet
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import requests
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import cl_predict as cp

console = cp.console
ELO_DIR = "elo_history"          # sparade Elo-listor per datum
RESULT_CSV = "backtest_results.csv"
K_FACTOR = 10.0                  # hur snabbt den rullande Elon reagerar


@dataclass
class Played:
    day: date
    home: str
    away: str
    hg: int
    ag: int
    neutral: bool
    stage: str

    @property
    def outcome(self) -> int:    # 0 = hemmaseger, 1 = oavgjort, 2 = bortaseger
        return 0 if self.hg > self.ag else 1 if self.hg == self.ag else 2


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def fetch_matches(api_key: str, season: int | None):
    """Spelade CL-matcher. Resultat efter 90 min (förlängning räknas inte)."""
    params = {"status": "FINISHED"}
    if season:
        params["season"] = season
    r = requests.get("https://api.football-data.org/v4/competitions/CL/matches",
                     headers={"X-Auth-Token": api_key}, params=params, timeout=20)
    if season and r.status_code in (400, 403, 404):
        console.print(f"[yellow]API:t ger inte säsong {season} på din nivå – "
                      f"använder innevarande säsong i stället.[/]")
        return fetch_matches(api_key, None)
    r.raise_for_status()

    out = []
    for m in r.json().get("matches", []):
        sc = m.get("score") or {}
        if sc.get("duration", "REGULAR") != "REGULAR" and sc.get("regularTime"):
            res = sc["regularTime"]
        else:
            res = sc.get("fullTime") or {}
        if res.get("home") is None:
            continue
        out.append(Played(
            day=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).date(),
            home=m["homeTeam"].get("shortName") or m["homeTeam"]["name"],
            away=m["awayTeam"].get("shortName") or m["awayTeam"]["name"],
            hg=res["home"], ag=res["away"],
            neutral=m.get("stage") == "FINAL",
            stage=m.get("stage") or "",
        ))
    label = season or (r.json().get("filters") or {}).get("season") or "innevarande"
    return sorted(out, key=lambda x: x.day), label


def demo_matches(n: int = 180):
    """Påhittade matcher där 'sanningen' är en något annorlunda modell än vår."""
    teams = list(cp.FALLBACK_ELO)
    start = date(2025, 9, 16)
    out = []
    for i in range(n):
        h, a = random.sample(teams, 2)
        diff = cp.FALLBACK_ELO[h] + 70 - cp.FALLBACK_ELO[a] + random.gauss(0, 60)
        lh = 1.45 * math.exp(0.0026 * diff / 2)
        la = 1.45 * math.exp(-0.0026 * diff / 2)
        out.append(Played(start + timedelta(days=i // 9 * 7), h, a,
                          poisson_sample(lh), poisson_sample(la), False, "LEAGUE_STAGE"))
    return out


def poisson_sample(lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p < limit:
            return k
        k += 1


# ----------------------------------------------------------------------------
# Elo utan att titta in i framtiden
# ----------------------------------------------------------------------------
_clubelo_down = False


def elo_on(day: date) -> dict | None:
    """Elo-listan från clubelo för ett visst datum (sparas lokalt)."""
    global _clubelo_down
    os.makedirs(ELO_DIR, exist_ok=True)
    path = os.path.join(ELO_DIR, f"{day.isoformat()}.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {row["Club"]: float(row["Elo"]) for row in csv.DictReader(f)}
    if _clubelo_down:
        return None
    try:
        r = requests.get(f"http://api.clubelo.com/{day.isoformat()}",
                         headers=cp.HEADERS, timeout=10)
        r.raise_for_status()
        if "Club" not in r.text[:200]:
            raise ValueError("oväntat svar")
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        return {row["Club"]: float(row["Elo"]) for row in csv.DictReader(io.StringIO(r.text))}
    except Exception as e:
        console.print(f"[dim]clubelo svarar inte ({e}) – byter till rullande Elo.[/]")
        _clubelo_down = True
        return None


def team_elo(name: str, table: dict, default: float = 1650.0) -> float:
    alias = cp.ALIASES.get(name, name)
    if alias in table:
        return table[alias]
    import difflib
    hit = difflib.get_close_matches(alias, table.keys(), n=1, cutoff=0.6)
    return table[hit[0]] if hit else default


def elo_update(R: dict, m: Played):
    exp_h = 1 / (1 + 10 ** (-(R[m.home] + (0 if m.neutral else cp.HOME_ADV_ELO) - R[m.away]) / 400))
    res = (1.0, 0.5, 0.0)[m.outcome]
    gd = abs(m.hg - m.ag)
    mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
    d = K_FACTOR * mult * (res - exp_h)
    R[m.home] += d
    R[m.away] -= d


# ----------------------------------------------------------------------------
# Modell & mått
# ----------------------------------------------------------------------------
def model_probs(elo_h: float, elo_a: float, neutral: bool):
    diff = elo_h + (0 if neutral else cp.HOME_ADV_ELO) - elo_a
    lh = cp.BASE_GOALS * math.exp(cp.ELO_TO_GOALS * diff / 2)
    la = cp.BASE_GOALS * math.exp(-cp.ELO_TO_GOALS * diff / 2)
    ph = pd = pa = 0.0
    best, best_p = (0, 0), 0.0
    for i in range(cp.MAX_GOALS + 1):
        for j in range(cp.MAX_GOALS + 1):
            p = cp.poisson(i, lh) * cp.poisson(j, la)
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p
            if p > best_p:
                best, best_p = (i, j), p
    t = ph + pd + pa
    return (ph / t, pd / t, pa / t), best


def rps(p, o: int) -> float:
    obs = [1.0 if k == o else 0.0 for k in range(3)]
    c1 = p[0] - obs[0]
    c2 = (p[0] + p[1]) - (obs[0] + obs[1])
    return 0.5 * (c1 ** 2 + c2 ** 2)


def brier(p, o: int) -> float:
    return sum((p[k] - (1.0 if k == o else 0.0)) ** 2 for k in range(3))


def logloss(p, o: int) -> float:
    return -math.log(max(p[o], 1e-12))


BASELINES = {
    "Slumpen (33/33/33)": (1 / 3, 1 / 3, 1 / 3),
    "Typisk fördelning (45/25/30)": (0.45, 0.25, 0.30),
}


# ----------------------------------------------------------------------------
# Visning
# ----------------------------------------------------------------------------
def metrics_table(scores: dict, n: int) -> Table:
    t = Table(title=f"Resultat på {n} matcher", title_style="bold",
              header_style="bold magenta", border_style="blue")
    t.add_column("Modell", no_wrap=True)
    t.add_column("RPS ↓", justify="right")
    t.add_column("Brier ↓", justify="right")
    t.add_column("Log loss ↓", justify="right")
    t.add_column("Träff 1X2 ↑", justify="right")
    best_rps = min(s["rps"] for s in scores.values())
    for name, s in scores.items():
        style = "bold green" if s["rps"] == best_rps else ""
        t.add_row(Text(name, style=style), Text(f"{s['rps']:.4f}", style=style),
                  f"{s['brier']:.4f}", f"{s['log']:.4f}", f"{s['acc']:.1%}")
    return t


def calibration_table(pairs) -> Table:
    """Om modellen säger 60 % – händer det då ungefär 60 % av gångerna?"""
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    t = Table(title="Kalibrering – stämmer procenten?", title_style="bold",
              header_style="bold magenta", border_style="blue")
    t.add_column("Modellen sa")
    t.add_column("Antal", justify="right")
    t.add_column("Snitt sagt", justify="right")
    t.add_column("Hände", justify="right")
    t.add_column("")
    for lo, hi in buckets:
        sel = [(p, o) for p, o in pairs if lo <= p < hi]
        if not sel:
            continue
        said = sum(p for p, _ in sel) / len(sel)
        happened = sum(o for _, o in sel) / len(sel)
        gap = happened - said
        mark = ("[green]✓ bra[/]" if abs(gap) < 0.05 else
                "[yellow]för försiktig[/]" if gap > 0 else "[red]för självsäker[/]")
        t.add_row(f"{lo:.0%}–{min(hi, 1):.0%}", str(len(sel)), f"{said:.0%}",
                  f"{happened:.0%}", mark)
    return t


def surprises_table(rows, k: int = 5) -> Table:
    t = Table(title="Största skrällarna (modellen trodde minst på det som hände)",
              title_style="bold", header_style="bold magenta", border_style="blue")
    t.add_column("Datum", style="dim")
    t.add_column("Match")
    t.add_column("Resultat", justify="center")
    t.add_column("Modellens chans", justify="right")
    for r in sorted(rows, key=lambda r: r["p_actual"])[:k]:
        t.add_row(r["date"], f"{r['home']} – {r['away']}", f"{r['hg']}–{r['ag']}",
                  Text(f"{r['p_actual']:.0%}", style="bold red"))
    return t


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Backtest av CL Predictor")
    ap.add_argument("--season", type=int, default=2025, help="säsongens startår (standard 2025)")
    ap.add_argument("--demo", action="store_true", help="påhittade matcher")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    console.print(Panel(Align.center(Text("BACKTEST – hur bra är modellen egentligen?",
                                          style="bold white")), border_style="magenta"))

    key = os.getenv("FOOTBALL_DATA_KEY") or cp.API_KEY
    if args.demo or not key:
        console.print("[yellow]Demo-läge – påhittade matcher.[/]")
        matches, label = demo_matches(), "demo"
        mode = "rullande"
    else:
        with console.status("[bold]Hämtar spelade matcher…", spinner="dots12"):
            matches, label = fetch_matches(key, args.season)
        mode = None
    if not matches:
        console.print("[red]Inga spelade matcher hittades.[/]")
        return
    console.print(f"Säsong: [bold]{label}[/] • [bold]{len(matches)}[/] spelade matcher\n")

    # Startvärden för Elo
    teams = {m.home for m in matches} | {m.away for m in matches}
    R = None
    if mode is None:
        first = elo_on(matches[0].day - timedelta(days=1))
        if first:
            mode = "clubelo"
        else:
            mode = "rullande"
            start = cp.FALLBACK_ELO
            if os.path.exists("elo_cache.csv"):
                with open("elo_cache.csv", encoding="utf-8") as f:
                    start = {row["Club"]: float(row["Elo"]) for row in csv.DictReader(f)}
            R = {t: team_elo(t, start) for t in teams}
    if R is None:
        R = {t: team_elo(t, cp.FALLBACK_ELO) for t in teams}

    console.print("[dim]Elo-läge: " + ("clubelo (dagen före varje match)" if mode == "clubelo"
                  else "rullande Elo (uppdateras efter varje match)") + "[/]")
    if mode == "rullande" and not args.demo:
        console.print("[dim]Obs: startvärdena är dagens/ungefärliga Elo, så de första "
                      "omgångarna kan bli något för bra.[/]")

    # Kör igenom matcherna i datumordning
    names = ["CL Predictor"] + list(BASELINES)
    sums = {n: {"rps": 0.0, "brier": 0.0, "log": 0.0, "acc": 0.0} for n in names}
    rows, cal_pairs, exact = [], [], 0
    with Progress(SpinnerColumn("dots12"), TextColumn("[bold]Tippar matcher utan facit"),
                  BarColumn(bar_width=40), TextColumn("{task.completed}/{task.total}"),
                  console=console, transient=True) as prog:
        task = prog.add_task("", total=len(matches))
        for m in matches:
            if mode == "clubelo":
                table = elo_on(m.day - timedelta(days=1)) or {}
                eh, ea = team_elo(m.home, table), team_elo(m.away, table)
            else:
                eh, ea = R[m.home], R[m.away]

            p, best = model_probs(eh, ea, m.neutral)
            o = m.outcome
            for name, probs in [("CL Predictor", p)] + list(BASELINES.items()):
                s = sums[name]
                s["rps"] += rps(probs, o)
                s["brier"] += brier(probs, o)
                s["log"] += logloss(probs, o)
                s["acc"] += 1.0 if max(range(3), key=lambda k: probs[k]) == o else 0.0
            exact += best == (m.hg, m.ag)
            for k in range(3):
                cal_pairs.append((p[k], 1.0 if k == o else 0.0))
            rows.append({"date": m.day.isoformat(), "home": m.home, "away": m.away,
                         "hg": m.hg, "ag": m.ag, "p_home": round(p[0], 4),
                         "p_draw": round(p[1], 4), "p_away": round(p[2], 4),
                         "tips": f"{best[0]}-{best[1]}", "p_actual": round(p[o], 4),
                         "rps": round(rps(p, o), 4)})

            if mode == "rullande":
                elo_update(R, m)          # lär sig av resultatet FÖRST efter tipset
            prog.advance(task)
            time.sleep(0.01)

    n = len(matches)
    scores = {name: {k: v / n for k, v in s.items()} for name, s in sums.items()}

    console.print(Rule("[bold cyan]📊 Resultat[/]"))
    console.print(metrics_table(scores, n))

    model, base = scores["CL Predictor"]["rps"], scores["Slumpen (33/33/33)"]["rps"]
    typ = scores["Typisk fördelning (45/25/30)"]["rps"]
    skill = (1 - model / base) * 100
    skill_typ = (1 - model / typ) * 100
    verdict = (f"Modellen har [bold]{skill:.1f} % lägre fel[/] (RPS) än slumpen och "
               f"[bold]{skill_typ:+.1f} %[/] jämfört med typisk fördelning.\n"
               f"Exakt rätt resultat: [bold]{exact}[/] av {n} ({exact / n:.0%}).")
    console.print(Panel(verdict, border_style="green" if skill_typ > 0 else "red"))

    console.print(calibration_table(cal_pairs))
    console.print(surprises_table(rows))

    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    console.print(f"\n[dim]Alla tips och resultat sparade i {RESULT_CSV} "
                  f"(öppna i Excel eller använd i README:n).[/]")
    console.print("[dim]Hur man läser: RPS 0 = perfekt, lägre är bättre. Fotboll är "
                  "svårtippat, så även de bästa modellerna hamnar sällan långt under 0.20.[/]")


if __name__ == "__main__":
    main()
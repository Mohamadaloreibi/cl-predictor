"""
CL Predictor – dagens Champions League-matcher i terminalen.

Kör:
  python cl_predict.py              # dagens riktiga matcher
  python cl_predict.py --demo       # exempeldata
  python cl_predict.py --date 2026-10-21
  python cl_predict.py --nosim      # hoppa över live-simuleringen
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()
TZ = ZoneInfo("Europe/Stockholm")


# ----------------------------------------------------------------------------
# API-nyckel – läses från .env (se .env.example), ALDRIG direkt i koden
# ----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
API_KEY = os.getenv("FOOTBALL_DATA_KEY", "")


# ----------------------------------------------------------------------------
# Modellparametrar
# ----------------------------------------------------------------------------
BASE_GOALS = 1.6       # snittmål per lag i CL (optimerat, steg 3)
ELO_TO_GOALS = 0.004   # hur hårt Elo-skillnaden slår på xG (optimerat, steg 3)
HOME_ADV_ELO = 75      # hemmafördel i Elo-poäng (optimerat, steg 3)
MAX_GOALS = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# football-data-namn -> clubelo-namn
ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "FC Bayern München": "Bayern", "PSG": "Paris SG", "Paris Saint-Germain": "Paris SG",
    "Internazionale": "Inter", "Atleti": "Atletico", "Atlético Madrid": "Atletico",
    "Barça": "Barcelona", "Sporting CP": "Sporting", "Club Brugge": "Brugge",
    "Red Bull Salzburg": "Salzburg", "Bodø/Glimt": "Bodoe/Glimt",
    "København": "FC Kobenhavn", "Fenerbahçe": "Fenerbahce", "PAE AEK": "AEK",
    "AEK Athens": "AEK", "Sl. Bratislava": "Slovan Bratislava", "Sabah FK": "Sabah",
}

# Ungefärliga Elo-värden, används bara om clubelo.com inte svarar
FALLBACK_ELO = {
    "Real Madrid": 1980, "Man City": 1990, "Liverpool": 2000, "Arsenal": 2000,
    "Bayern": 1990, "Barcelona": 1980, "Paris SG": 1990, "Inter": 1950,
    "Chelsea": 1920, "Atletico": 1880, "Leverkusen": 1880, "Dortmund": 1840,
    "Napoli": 1860, "Juventus": 1830, "Milan": 1820, "Atalanta": 1830,
    "Tottenham": 1830, "Newcastle": 1860, "Aston Villa": 1840, "Benfica": 1820,
    "Sporting": 1820, "Porto": 1790, "PSV": 1800, "RB Leipzig": 1800,
    "Frankfurt": 1790, "Marseille": 1800, "Monaco": 1780, "Galatasaray": 1760,
    "Brugge": 1740, "Feyenoord": 1740, "Ajax": 1740, "Bodoe/Glimt": 1750,
    "Celtic": 1700, "Olympiakos": 1700, "FC Kobenhavn": 1680, "Slavia Praha": 1700,
    "Fenerbahce": 1770, "Stuttgart": 1810, "Villarreal": 1820, "Roma": 1810,
    "AEK": 1710, "Slovan Bratislava": 1640, "Sabah": 1580,
}


@dataclass
class Match:
    home: str
    away: str
    kickoff: datetime
    home_elo: float | None = None
    away_elo: float | None = None
    home_id: int | None = None
    away_id: int | None = None

@dataclass
class Prediction:
    p_home: float
    p_draw: float
    p_away: float
    xg_home: float
    xg_away: float
    score: tuple[int, int]
    score_p: float
    top_scores: list = field(default_factory=list)   # [(h, a, p), ...]
    over15: float = 0.0
    over25: float = 0.0
    over35: float = 0.0
    btts: float = 0.0
    cs_home: float = 0.0   # hemmalaget håller nollan
    cs_away: float = 0.0   # bortalaget håller nollan


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def fetch_fixtures(day: date, api_key: str) -> list[Match]:
    url = "https://api.football-data.org/v4/competitions/CL/matches"
    r = requests.get(url, headers={"X-Auth-Token": api_key},
                     params={"dateFrom": day.isoformat(), "dateTo": day.isoformat()},
                     timeout=15)
    r.raise_for_status()
    out = []
    for m in r.json().get("matches", []):
        ko = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(TZ)
        out.append(Match(
            home=m["homeTeam"].get("shortName") or m["homeTeam"]["name"],
            away=m["awayTeam"].get("shortName") or m["awayTeam"]["name"],
            kickoff=ko,
            home_id=m["homeTeam"].get("id"),
            away_id=m["awayTeam"].get("id"),
        ))
    return sorted(out, key=lambda x: x.kickoff)

SQUAD_CACHE = "squads_cache.json"


def position_weight(pos: str | None) -> float:
    """Hur stor chans en spelare har att göra mål beroende på position."""
    p = (pos or "").lower()
    if "goalkeeper" in p:
        return 0.0
    if any(k in p for k in ("offence", "forward", "striker", "winger")):
        return 6.0
    if "attacking midfield" in p:
        return 4.0
    if "midfield" in p:
        return 2.0
    if any(k in p for k in ("defence", "back")):
        return 0.6
    return 1.0


def fetch_squads(matches: list[Match], api_key: str, prog=None, task=None) -> dict[int, list]:
    """Hämtar trupper från football-data. Sparas i squads_cache.json i 7 dagar.
    Gratisnivån tillåter 10 anrop/minut, så första körningen tar en stund."""
    cache = {}
    if os.path.exists(SQUAD_CACHE):
        try:
            with open(SQUAD_CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    ids = sorted({i for m in matches for i in (m.home_id, m.away_id) if i})
    if prog is not None:
        prog.update(task, total=len(ids))

    now = time.time()
    squads = {}
    for n, tid in enumerate(ids):
        entry = cache.get(str(tid))
        if entry and now - entry["ts"] < 7 * 86400:
            squads[tid] = entry["players"]
        else:
            for attempt in range(2):
                try:
                    r = requests.get(f"https://api.football-data.org/v4/teams/{tid}",
                                     headers={"X-Auth-Token": api_key}, timeout=15)
                    if r.status_code == 429:          # för många anrop – vänta en minut
                        if prog is not None:
                            prog.update(task, description="Väntar på API-gränsen (60 s)…")
                        time.sleep(61)
                        if prog is not None:
                            prog.update(task, description="Hämtar trupper…")
                        continue
                    r.raise_for_status()
                    players = [[pl["name"], position_weight(pl.get("position"))]
                               for pl in r.json().get("squad", [])]
                    players = [pl for pl in players if pl[1] > 0]
                    squads[tid] = players
                    cache[str(tid)] = {"ts": now, "players": players}
                    break
                except Exception as e:
                    console.print(f"[dim]Trupp {tid}: {e}[/]")
                    break
            time.sleep(6.5)                            # håll oss under 10 anrop/minut
        if prog is not None:
            prog.update(task, completed=n + 1)

    with open(SQUAD_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    return squads


def demo_squad(team: str) -> list:
    pos = [("Anfallare", 6.0)] * 3 + [("Mittfältare", 2.0)] * 4 + [("Back", 0.6)] * 4
    return [[f"{kind}{i + 1}", w] for i, (kind, w) in enumerate(pos)]


def short(name: str) -> str:
    """Efternamn för listan (fullt namn visas i MÅL-rutan)."""
    parts = name.split()
    return parts[-1] if len(parts) > 1 and not parts[-1].isdigit() else name


def pick_scorer(players: list | None, team: str) -> str:
    if not players:
        return team
    names = [p[0] for p in players]
    weights = [p[1] for p in players]
    return random.choices(names, weights=weights, k=1)[0]

def fetch_elo(day: date) -> dict[str, float]:
    """Hämtar Elo från clubelo. Annars sparad kopia (elo_cache.csv), annars reservlista."""
    cache = "elo_cache.csv"
    for d in (day, day - timedelta(days=1)):
        url = f"http://api.clubelo.com/{d.isoformat()}"
        try:
            r = requests.get(url, timeout=10, headers=HEADERS)
            r.raise_for_status()
            if "Club" not in r.text[:200]:
                raise ValueError("oväntat svar")
            with open(cache, "w", encoding="utf-8") as f:
                f.write(r.text)
            return {row["Club"]: float(row["Elo"]) for row in csv.DictReader(io.StringIO(r.text))}
        except Exception as e:
            console.print(f"[dim]clubelo {d}: {e}[/]")

    if os.path.exists(cache):
        console.print("[yellow]clubelo.com svarar inte – använder sparade Elo-värden.[/]")
        with open(cache, encoding="utf-8") as f:
            return {row["Club"]: float(row["Elo"]) for row in csv.DictReader(f)}

    console.print("[yellow]clubelo.com svarar inte – använder ungefärliga reserv-Elo.[/]")
    return FALLBACK_ELO


def lookup_elo(team: str, elo: dict[str, float]) -> float | None:
    name = ALIASES.get(team, team)
    if name in elo:
        return elo[name]
    hit = difflib.get_close_matches(name, elo.keys(), n=1, cutoff=0.6)
    if hit:
        return elo[hit[0]]
    if elo is FALLBACK_ELO:
        console.print(f"[dim]{team} saknas i reservlistan – sätter 1680.[/]")
        return 1680.0
    return None


def demo_data(day: date) -> list[Match]:
    def ko(h, m):
        return datetime(day.year, day.month, day.day, h, m, tzinfo=TZ)
    return [
        Match("Real Madrid", "Juventus", ko(18, 45), 1985, 1820),
        Match("Bodø/Glimt", "Monaco", ko(18, 45), 1740, 1790),
        Match("Arsenal", "Atleti", ko(21, 0), 2010, 1880),
        Match("Bayern", "Club Brugge", ko(21, 0), 1990, 1700),
        Match("PSV", "Napoli", ko(21, 0), 1800, 1860),
        Match("Liverpool", "Inter", ko(21, 0), 1995, 1960),
    ]


# ----------------------------------------------------------------------------
# Modell
# ----------------------------------------------------------------------------
def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def predict(m: Match) -> Prediction:
    diff = (m.home_elo + HOME_ADV_ELO) - m.away_elo
    xg_h = BASE_GOALS * math.exp(ELO_TO_GOALS * diff / 2)
    xg_a = BASE_GOALS * math.exp(-ELO_TO_GOALS * diff / 2)

    ph = pd = pa = 0.0
    o15 = o25 = o35 = btts = cs_h = cs_a = 0.0
    scores = []
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson(i, xg_h) * poisson(j, xg_a)
            scores.append((i, j, p))
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p
            if i + j > 1.5: o15 += p
            if i + j > 2.5: o25 += p
            if i + j > 3.5: o35 += p
            if i > 0 and j > 0: btts += p
            if j == 0: cs_h += p
            if i == 0: cs_a += p

    total = ph + pd + pa
    scores.sort(key=lambda s: s[2], reverse=True)
    best = scores[0]
    return Prediction(
        ph / total, pd / total, pa / total, xg_h, xg_a,
        (best[0], best[1]), best[2], scores[:5],
        o15 / total, o25 / total, o35 / total, btts / total, cs_h / total, cs_a / total,
    )


def verdict(m: Match, p: Prediction) -> str:
    """En kort text som beskriver matchbilden."""
    top = max(p.p_home, p.p_away)
    fav = m.home if p.p_home >= p.p_away else m.away
    if top >= 0.60:
        txt = f"Klar favorit: {fav}"
    elif top >= 0.45:
        txt = f"{fav} något bättre"
    else:
        txt = "Jämn match – allt kan hända"
    if p.over25 >= 0.58:
        txt += " • väntas bli målrik"
    elif p.over25 <= 0.42:
        txt += " • väntas bli tight"
    return txt


# ----------------------------------------------------------------------------
# Animationer & visning
# ----------------------------------------------------------------------------
PITCH_W, PITCH_H = 26, 9


def pitch_frame(t: int) -> Text:
    period_x, period_y = 2 * (PITCH_W - 3), 2 * (PITCH_H - 3)
    bx = t % period_x
    bx = bx if bx < PITCH_W - 3 else period_x - bx
    by = (t // 2) % period_y
    by = by if by < PITCH_H - 3 else period_y - by
    bx, by = bx + 1, by + 1
    rows = []
    for y in range(PITCH_H):
        line = []
        for x in range(PITCH_W):
            if y in (0, PITCH_H - 1):
                ch = "─"
            elif x in (0, PITCH_W - 1):
                ch = "┃" if PITCH_H // 2 - 1 <= y <= PITCH_H // 2 + 1 else "│"
            elif x == PITCH_W // 2:
                ch = "┊"
            else:
                ch = " "
            if (x, y) == (bx, by):
                ch = "●"
            line.append(ch)
        rows.append("".join(line))
    txt = Text("\n".join(rows), style="green")
    txt.highlight_regex("●", "bold white")
    return txt


def ticker(t: int, msg: str) -> Text:
    width = 24
    s = (msg + "   •   ") * 3
    start = t % (len(msg) + 7)
    return Text(s[start:start + width], style="bold yellow")


def bar(p: float, width: int = 12, color: str = "cyan") -> Text:
    filled = round(p * width)
    return Text("█" * filled + "░" * (width - filled), style=color)


def pct(p: float) -> str:
    return f"{p:.0%}"


def build_table(rows, reveal: int) -> Table:
    tbl = Table(expand=True, header_style="bold magenta", border_style="blue")
    tbl.add_column("Tid", style="dim", width=5)
    tbl.add_column("Match", ratio=1, overflow="ellipsis", no_wrap=True)
    tbl.add_column("1", justify="right")
    tbl.add_column("X", justify="right")
    tbl.add_column("2", justify="right")
    tbl.add_column("Ö2.5", justify="right")
    tbl.add_column("BLGM", justify="right")
    tbl.add_column("Tips", justify="center", style="bold green")

    for idx, (m, p) in enumerate(rows):
        match_txt = f"{m.home} – {m.away}"
        if idx >= reveal:
            dots = "." * ((int(time.time() * 3) + idx) % 4)
            tbl.add_row(m.kickoff.strftime("%H:%M"), match_txt, "", f"[dim]räknar{dots}[/]",
                        "", "", "", "")
            continue
        fav = max((p.p_home, "1"), (p.p_draw, "X"), (p.p_away, "2"))[1]

        def cell(val, key):
            return Text(pct(val), style="bold white on dark_green" if key == fav else "")

        tbl.add_row(m.kickoff.strftime("%H:%M"), match_txt,
                    cell(p.p_home, "1"), cell(p.p_draw, "X"), cell(p.p_away, "2"),
                    pct(p.over25), pct(p.btts), f"{p.score[0]}–{p.score[1]}")
    return tbl


def side_panel(t: int, rows, reveal: int) -> Panel:
    done = rows[:reveal]
    if done:
        m, p = done[-1]
        focus = Group(
            Text(f"\n{m.home}", style="bold"), bar(p.p_home, color="green"),
            Text("Oavgjort", style="bold"), bar(p.p_draw, color="yellow"),
            Text(m.away, style="bold"), bar(p.p_away, color="red"),
        )
    else:
        focus = Text("\nVäntar på första matchen…", style="dim")
    return Panel(Group(Align.center(pitch_frame(t)),
                       Align.center(ticker(t, "UEFA CHAMPIONS LEAGUE")), focus),
                 title="⚽ Live", border_style="green")


def match_card(m: Match, p: Prediction) -> Panel:
    """Detaljerad statistik för en match."""
    # Vinstchans
    win = Table.grid(padding=(0, 1))
    win.add_column(width=14, no_wrap=True, overflow="ellipsis")
    win.add_column()
    win.add_column(justify="right")
    win.add_row(m.home, bar(p.p_home, 14, "green"), pct(p.p_home))
    win.add_row("Oavgjort", bar(p.p_draw, 14, "yellow"), pct(p.p_draw))
    win.add_row(m.away, bar(p.p_away, 14, "red"), pct(p.p_away))

    # Troligaste resultat
    res = Table(title="Troligaste resultat", title_style="bold", box=None,
                show_header=False, padding=(0, 1))
    res.add_column(justify="center")
    res.add_column()
    res.add_column(justify="right")
    for k, (h, a, prob) in enumerate(p.top_scores):
        style = "bold green" if k == 0 else ""
        res.add_row(Text(f"{h}–{a}", style=style), bar(prob / p.top_scores[0][2] * 0.6, 8, "cyan"),
                    Text(f"{prob:.1%}", style=style))

    # Statistik
    st = Table(title="Statistik", title_style="bold", box=None, show_header=False, padding=(0, 1))
    st.add_column()
    st.add_column(justify="right")
    st.add_row("Förväntade mål (xG)", f"{p.xg_home:.2f} – {p.xg_away:.2f}")
    st.add_row("Elo", f"{m.home_elo:.0f} – {m.away_elo:.0f}")
    st.add_row("Över 1.5 mål", pct(p.over15))
    st.add_row("Över 2.5 mål", pct(p.over25))
    st.add_row("Över 3.5 mål", pct(p.over35))
    st.add_row("Båda lagen gör mål", pct(p.btts))
    st.add_row(f"{m.home} håller nollan", pct(p.cs_home))
    st.add_row(f"{m.away} håller nollan", pct(p.cs_away))

    body = Group(
        win,
        Text(""),
        Columns([res, st], padding=(0, 4)),
        Text(""),
        Text(f"📋 {verdict(m, p)}", style="bold yellow"),
    )
    title = f"[bold]{m.home} – {m.away}[/]  [dim]{m.kickoff.strftime('%H:%M')}[/]"
    sub = f"Tips: [bold green]{p.score[0]}–{p.score[1]}[/]"
    return Panel(body, title=title, subtitle=sub, border_style="cyan", padding=(1, 2))


def simulate_round(rows, squads: dict, seconds: float = 14.0):
    """Spelar upp alla matcher minut för minut. Mål slumpas utifrån xG,
    målskytten slumpas ur truppen (anfallare oftast)."""
    console.print(Rule("[bold magenta]⚡ Simulerad omgång – en möjlig utgång[/]"))
    state = [{"h": 0, "a": 0, "goals": []} for _ in rows]
    popup = {"text": None, "until": 0}

    def board(minute: int):
        t = Table(expand=True, box=None, padding=(0, 1))
        t.add_column("Hemma", justify="right", ratio=2, overflow="fold")
        t.add_column("", justify="center", width=7)
        t.add_column("Borta", ratio=2, overflow="fold")
        for (m, _), s in zip(rows, state):
            t.add_row(m.home, Text(f"{s['h']} – {s['a']}", style="bold white on blue"), m.away)
            if s["goals"]:
                home_g = ", ".join(f"{short(g[1])} {g[0]}'" for g in s["goals"] if g[2] == "h")
                away_g = ", ".join(f"{g[0]}' {short(g[1])}" for g in s["goals"] if g[2] == "a")
                t.add_row(Text(home_g, style="dim"), "", Text(away_g, style="dim"))
        clock = "SLUT" if minute >= 90 else f"{minute}'"
        prog = bar(min(minute, 90) / 90, 30, "magenta")
        parts = [Align.center(Text(clock, style="bold yellow")), Align.center(prog), Text("")]
        if popup["text"] and minute <= popup["until"]:
            flash = "bold black on yellow" if minute % 2 else "bold yellow on red"
            parts.append(Align.center(Panel(Text(popup["text"], style=flash, justify="center"),
                                            border_style="yellow", padding=(0, 3))))
        parts.append(t)
        return Panel(Group(*parts), border_style="magenta", title="Live-simulering")

    with Live(board(0), console=console, refresh_per_second=20) as live:
        for minute in range(1, 91):
            for (m, p), s in zip(rows, state):
                for side, xg, team, tid in (("h", p.xg_home, m.home, m.home_id),
                                            ("a", p.xg_away, m.away, m.away_id)):
                    if random.random() < xg / 90:
                        s[side] += 1
                        players = squads.get(tid) if tid else squads.get(team)
                        scorer = pick_scorer(players, team)
                        s["goals"].append((minute, scorer, side))
                        popup["text"] = (f"⚽  MÅL!  {minute}'  {scorer}  ({team})\n"
                                         f"{m.home} {s['h']} – {s['a']} {m.away}")
                        popup["until"] = minute + 8
            live.update(board(minute))
            # sakta ner lite när det precis blivit mål
            time.sleep(seconds / 90 * (2.5 if popup["until"] - 8 == minute else 1))
        popup["text"] = None
        live.update(board(90))

def intro():
    frames = ["⚽      ", " ⚽     ", "  ⚽    ", "   ⚽   ", "    ⚽  ", "     ⚽ ", "      🥅"]
    for f in frames * 2:
        console.print(f"[bold green]{f}[/]  [bold]CL PREDICTOR[/]", end="\r")
        time.sleep(0.08)
    console.print(" " * 40, end="\r")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Prediktioner för dagens CL-matcher")
    ap.add_argument("--date", help="YYYY-MM-DD (standard: idag)")
    ap.add_argument("--demo", action="store_true", help="kör med exempeldata")
    ap.add_argument("--nosim", action="store_true", help="hoppa över live-simuleringen")
    args = ap.parse_args()

    day = date.fromisoformat(args.date) if args.date else datetime.now(TZ).date()
    key = os.getenv("FOOTBALL_DATA_KEY") or API_KEY
    demo = args.demo or not key

    intro()

    with Progress(SpinnerColumn("dots12"), TextColumn("[bold]{task.description}"),
                  BarColumn(), console=console, transient=True) as prog:
        t1 = prog.add_task("Hämtar dagens matcher…", total=1)
        if demo:
            if not args.demo:
                console.print("[yellow]Ingen API-nyckel satt – kör demo-läge.[/]")
            matches = demo_data(day)
            time.sleep(0.6)
        else:
            matches = fetch_fixtures(day, key)
        prog.update(t1, completed=1)

        t2 = prog.add_task("Hämtar Elo-ratingar…", total=1)
        if not demo:
            elo = fetch_elo(day)
            for m in matches:
                m.home_elo = lookup_elo(m.home, elo)
                m.away_elo = lookup_elo(m.away, elo)
        else:
            time.sleep(0.6)
        prog.update(t2, completed=1)


        squads: dict = {}
        if not args.nosim:
            t3 = prog.add_task("Hämtar trupper…", total=1)
            if demo:
                for m in matches:
                    squads[m.home] = demo_squad(m.home)
                    squads[m.away] = demo_squad(m.away)
                prog.update(t3, completed=1)
            else:
                squads = fetch_squads(matches, key, prog, t3)



    if not matches:
        console.print(Panel(f"Inga CL-matcher {day.isoformat()} 🤷", border_style="red"))
        return

    rows = []
    for m in matches:
        if m.home_elo is None or m.away_elo is None:
            console.print(f"[red]Hittade inte Elo för "
                          f"{m.home if m.home_elo is None else m.away} – hoppar över.[/]")
            continue
        rows.append((m, predict(m)))

    # 1) Översikt med animation
    layout = Layout()
    layout.split_column(Layout(name="head", size=3), Layout(name="body"))
    layout["body"].split_row(Layout(name="table", ratio=2), Layout(name="side", size=34))
    layout["head"].update(Panel(
        Align.center(Text(f"CHAMPIONS LEAGUE  •  {day.strftime('%A %d %B %Y')}"
                          + ("  •  DEMO" if demo else ""), style="bold white")),
        border_style="magenta"))

    reveal, t = 0, 0
    with Live(layout, console=console, refresh_per_second=20):
        end_t = len(rows) * 20 + 40
        while t < end_t:
            if t % 20 == 0 and reveal < len(rows):
                reveal += 1
            layout["table"].update(Panel(build_table(rows, reveal), title="Prediktioner",
                                         border_style="blue"))
            layout["side"].update(side_panel(t, rows, reveal))
            time.sleep(0.05)
            t += 1

    # 2) Matchkort med statistik
    console.print(Rule("[bold cyan]📊 Matchkort[/]"))
    for m, p in rows:
        console.print(match_card(m, p))
        time.sleep(0.3)

    # 3) Simulerad omgång
    if not args.nosim:
        simulate_round(rows,squads)

    console.print("[dim]Modell: Elo → xG → Poisson. BLGM = båda lagen gör mål. "
                  "Underhållning, inte spelråd.[/]")


if __name__ == "__main__":
    main()
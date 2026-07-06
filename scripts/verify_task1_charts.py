#!/usr/bin/env python3
"""Rendering-truth harness for Task 1 chart generation.

Standalone — NOT wired into the app. Calls the exact same generation path the
app uses (backend.main.generate_chart_svg) with injected representative data,
then asserts STRUCTURAL truth by parsing the SVG, not by trusting the string
validator alone.

Run from repo root:  python3 scripts/verify_task1_charts.py
Requires backend/.env with ANTHROPIC_API_KEY (loaded by main.py's load_dotenv).
Makes real API calls (~20 Sonnet calls) — cost is deliberate.

PNG rendering: cairosvg needs system libcairo, absent on this machine, so the
harness falls back to parsed-geometry assertions and writes raw .svg files to
scripts/out/ for human eyeballing in a browser.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)

# main.py calls load_dotenv() which searches from cwd — run as if from backend/.
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

# STUB (disclosed): the local .env's SUPABASE_SERVICE_KEY is rejected by
# supabase-py at import time (SupabaseException: Invalid API key). This harness
# never touches the DB, so blank the two vars for THIS process only; main.py
# then takes its supported supabase_admin=None path. load_dotenv(override=False)
# does not overwrite pre-existing (even empty) env vars.
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""

from main import generate_chart_svg, _validate_chart_svg  # noqa: E402

try:
    import cairosvg  # noqa: F401
    HAVE_PNG = True
except Exception:
    HAVE_PNG = False


# ── representative cases (5 per served subtype) ──────────────────────────────
CASES = {
    "bar_chart": [
        {"title": "Government spending by sector in 2020",
         "desc": "Sector | Value\nEducation | 62\nHealth | 48\nTransport | 35\nHousing | 27"},
        {"title": "Weekly hours of internet use by age group",
         "desc": "Age group | Hours\n16-24 | 32\n25-34 | 28\n35-44 | 22\n45-54 | 15\n55-64 | 9"},
        {"title": "Recycling rates in five countries in 2019",
         "desc": "Country | Rate\nGermany | 66\nAustria | 58\nSouth Korea | 54\nWales | 52\nSwitzerland | 50"},
        {"title": "Urban and rural populations from 2000 to 2010", "grouped": True,
         "desc": "Year | Urban | Rural\n2000 | 45 | 30\n2005 | 52 | 38\n2010 | 61 | 44"},
        {"title": "Energy consumption by source from 1990 to 2020", "grouped": True,
         "desc": "Year | Coal | Gas | Renewables\n1990 | 55 | 30 | 5\n2000 | 45 | 38 | 10\n2010 | 35 | 40 | 18\n2020 | 20 | 42 | 30"},
    ],
    "line_graph": [
        {"title": "Average house prices from 2000 to 2015",
         "desc": "Year | Value\n2000 | 22\n2005 | 34\n2010 | 41\n2015 | 58"},
        {"title": "Annual visitors to a museum from 2010 to 2018",
         "desc": "Year | Visitors\n2010 | 15\n2012 | 24\n2014 | 31\n2016 | 45\n2018 | 52"},
        {"title": "Broadband subscriptions from 2005 to 2020",
         "desc": "Year | Subscriptions\n2005 | 12\n2008 | 25\n2011 | 38\n2014 | 51\n2017 | 63\n2020 | 71"},
        {"title": "Urban and rural literacy rates from 1990 to 2020",
         "desc": "Year | Urban | Rural\n1990 | 62 | 41\n2000 | 71 | 52\n2010 | 82 | 63\n2020 | 91 | 74"},
        {"title": "Electricity generation by source from 2000 to 2015",
         "desc": "Year | Coal | Gas | Wind\n2000 | 48 | 30 | 5\n2005 | 42 | 35 | 11\n2010 | 33 | 38 | 19\n2015 | 25 | 40 | 28"},
    ],
    "pie_chart": [
        {"title": "Household expenditure by category in 2022",
         "desc": "Category | Share\nHousing | 34\nTransport | 26\nFood | 22\nLeisure | 18"},
        {"title": "Sources of electricity in 2021",
         "desc": "Source | Share\nGas | 38\nCoal | 24\nNuclear | 17\nWind | 13\nSolar | 8"},
        {"title": "Modes of transport to work in 2019",
         "desc": "Mode | Share\nCar | 45\nBus | 25\nBicycle | 18\nWalking | 12"},
        {"title": "Household spending in 2015 and 2023", "two_period": True,
         "desc": "Sector | 2015 | 2023\nHousing | 30 | 38\nTransport | 28 | 22\nFood | 24 | 21\nLeisure | 18 | 19"},
        {"title": "Energy use by sector in 2000 and 2020", "two_period": True,
         "desc": "Sector | 2000 | 2020\nIndustry | 40 | 30\nTransport | 30 | 33\nHomes | 20 | 24\nServices | 10 | 13"},
    ],
    "table": [
        {"title": "Student enrolments by faculty from 2000 to 2010",
         "desc": "Year | Arts | Science | Law\n2000 | 45 | 30 | 12\n2005 | 52 | 28 | 15\n2010 | 61 | 24 | 18"},
        {"title": "Average rainfall in four cities by season",
         "desc": "City | Spring | Summer | Autumn | Winter\nLondon | 42 | 38 | 55 | 61\nParis | 48 | 45 | 52 | 50\nRome | 55 | 22 | 68 | 71\nMadrid | 38 | 12 | 44 | 40"},
        {"title": "Employment by industry in 1995 and 2015",
         "desc": "Industry | 1995 | 2015\nAgriculture | 22 | 9\nManufacturing | 31 | 18\nServices | 47 | 73"},
        {"title": "Tourist arrivals in five countries in 2018",
         "desc": "Country | Arrivals | Growth\nFrance | 89 | 3\nSpain | 83 | 4\nUSA | 80 | 2\nChina | 63 | 6\nItaly | 62 | 5"},
        {"title": "Water consumption by use from 1990 to 2020",
         "desc": "Year | Agriculture | Industry | Domestic\n1990 | 65 | 22 | 13\n2000 | 62 | 24 | 14\n2010 | 58 | 26 | 16\n2020 | 54 | 28 | 18"},
    ],
}

SVGNS = "{http://www.w3.org/2000/svg}"


def _local(tag):
    return tag.split("}")[-1]


def _elems(root, name):
    return [e for e in root.iter() if _local(e.tag) == name]


def _rows_cols(desc):
    lines = [ln for ln in desc.splitlines() if "|" in ln]
    rows = len(lines) - 1
    cols = len(lines[0].split("|"))
    return rows, cols


def _arc_paths(root):
    """(d, rx, ry, start_cx, start_cy) for every <path> whose d has an arc."""
    out = []
    for p in _elems(root, "path"):
        d = p.get("d") or ""
        arc = re.search(r"[Aa]\s*([\d.]+)[\s,]+([\d.]+)", d)
        if not arc:
            continue
        m = re.match(r"\s*[Mm]\s*([\d.-]+)[\s,]+([\d.-]+)", d)
        cx, cy = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        out.append((d, float(arc.group(1)), float(arc.group(2)), cx, cy))
    return out


def check_pie(root, case):
    arcs = _arc_paths(root)
    if _elems(root, "ellipse"):
        return "pie: found <ellipse>"
    if len(arcs) < 2:
        return f"pie: only {len(arcs)} arc <path> elements, expected >=2"
    bad = [(rx, ry) for _, rx, ry, _, _ in arcs if abs(rx - ry) > 0.01]
    if bad:
        return f"pie: non-circular arc rx!=ry {bad[:3]}"
    centres = {(round(cx), round(cy)) for _, _, _, cx, cy in arcs if cx is not None}
    radii = {round(rx) for _, rx, _, _, _ in arcs}
    if case.get("two_period"):
        if len(centres) != 2:
            return f"pie two-period: {len(centres)} arc centres {sorted(centres)}, expected exactly 2"
        if len(radii) != 1:
            return f"pie two-period: unequal radii across pies {sorted(radii)}"
    else:
        if len(centres) != 1:
            return f"pie single: {len(centres)} arc centres {sorted(centres)}, expected exactly 1"
    return None


def check_bar(root, case):
    rows, cols = _rows_cols(case["desc"])
    expected = rows * (cols - 1)
    bars = []
    for r in _elems(root, "rect"):
        try:
            x = float(r.get("x", 0)); y = float(r.get("y", 0))
            w = float(r.get("width", 0)); h = float(r.get("height", 0))
        except ValueError:
            continue
        if w >= 590:  # background
            continue
        if abs((y + h) - 340) <= 1 and x >= 60 and h > 2:
            bars.append(r)
    if len(bars) != expected:
        return f"bar: {len(bars)} baseline-touching bars, expected {expected}"
    return None


def check_line(root, case):
    rows, _ = _rows_cols(case["desc"])
    circles = _elems(root, "circle")
    for c in circles:
        cx = float(c.get("cx", -1)); cy = float(c.get("cy", -1))
        if not (69 <= cx <= 561 and 39 <= cy <= 341):
            return f"line: marker ({cx},{cy}) outside chart area"
    has_series = bool(_elems(root, "polyline")) or any(
        "L" in (p.get("d") or "").upper() for p in _elems(root, "path"))
    if not has_series:
        return "line: no <polyline> and no line-to <path> series"
    if len(circles) < rows:
        return f"line: {len(circles)} markers, expected >= {rows}"
    return None


def check_table(root, case):
    rows, cols = _rows_cols(case["desc"])
    if _elems(root, "ellipse"):
        return "table: found <ellipse>"
    if _arc_paths(root):
        return "table: found arc <path> (drawn as pie)"
    n_text = len(_elems(root, "text"))
    if n_text < rows * cols:
        return f"table: {n_text} <text> nodes, expected >= {rows * cols}"
    return None


CHECKS = {"pie_chart": check_pie, "bar_chart": check_bar,
          "line_graph": check_line, "table": check_table}


def run_case(subtype, i, case):
    readable = subtype.replace("_", " ")
    prompt = f"The {readable} below shows {case['title'][0].lower() + case['title'][1:]}."
    try:
        svg = generate_chart_svg(subtype, case["desc"], prompt, chart_title=case["title"])
        if not svg:
            return (subtype, i, False, "generate_chart_svg returned None (validator rejected both attempts)")
        # (a) validator with correct subtype
        ok, reason = _validate_chart_svg(svg, case["title"],
                                         [ln.split("|")[0].strip() for ln in case["desc"].splitlines()[1:]],
                                         subtype=subtype, chart_description=case["desc"])
        if not ok:
            return (subtype, i, False, f"validator: {reason}")
        # (b) structural truth
        root = ET.fromstring(svg)
        fail = CHECKS[subtype](root, case)
        # (c) artifact for human eyeballing
        ext_path = os.path.join(OUT_DIR, f"{subtype}_{i}.svg")
        with open(ext_path, "w") as f:
            f.write(svg)
        if HAVE_PNG:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(),
                             write_to=os.path.join(OUT_DIR, f"{subtype}_{i}.png"))
        if fail:
            return (subtype, i, False, fail)
        return (subtype, i, True, "")
    except Exception as exc:
        return (subtype, i, False, f"exception: {type(exc).__name__}: {exc}")


def main():
    jobs = [(st, i, c) for st, cases in CASES.items() for i, c in enumerate(cases)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: run_case(*j), jobs))

    summary = {}
    for st, i, ok, reason in results:
        summary.setdefault(st, {"pass": 0, "fails": []})
        if ok:
            summary[st]["pass"] += 1
        else:
            summary[st]["fails"].append(f"[{i}] {reason}")

    print()
    print(f"{'subtype':<12} | passed/5 | first_failure_reason")
    print("-" * 90)
    frozen = []
    for st in CASES:
        s = summary[st]
        first = s["fails"][0] if s["fails"] else "-"
        print(f"{st:<12} | {s['pass']}/5      | {first}")
        if s["pass"] < 5:
            frozen.append((st, first))
    print()
    print(f"PNG rendering: {'yes' if HAVE_PNG else 'NO — libcairo absent; geometry-only asserts, raw .svg written to scripts/out/ for eyeballing'}")
    if not frozen:
        print("SERVED 4/4 at 5/5: YES")
    else:
        detail = ", ".join(f"{st}: {r}" for st, r in frozen)
        print(f"SERVED 4/4 at 5/5: NO — frozen: {[st for st, _ in frozen]}, first_failure: {detail}")
    return 0 if not frozen else 1


if __name__ == "__main__":
    sys.exit(main())

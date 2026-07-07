#!/usr/bin/env python3
"""Rendering-truth harness for Task 1 chart generation.

Standalone — NOT wired into the app. Calls the exact same generation path the
app uses (backend.main.generate_chart_svg) with injected representative data,
then asserts STRUCTURAL truth by parsing the SVG, not by trusting the string
validator alone.

Run from repo root:  python3 scripts/verify_task1_charts.py
Requires backend/.env with ANTHROPIC_API_KEY (loaded by main.py's load_dotenv).
Makes real API calls (~24 Sonnet calls, 6 per subtype) — cost is deliberate.

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

# Load backend/.env BEFORE importing main, so main.py's module-level
# anthropic_client is constructed with the real key already in os.environ.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(BACKEND_DIR, ".env"))
_key = os.environ.get("ANTHROPIC_API_KEY")
print(f"ANTHROPIC_API_KEY loaded: {'yes (redacted)' if _key else 'NO — aborting'}")
if not _key:
    sys.exit(2)

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
        {"title": "Average commute time in five cities in 2023",
         "desc": "City | Minutes\nLondon | 46\nTokyo | 42\nNew York | 40\nSydney | 35\nBerlin | 30"},
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
        {"title": "Domestic and international air passengers from 2000 to 2020",
         "desc": "Year | Domestic | International\n2000 | 31 | 18\n2005 | 38 | 25\n2010 | 44 | 34\n2015 | 52 | 45\n2020 | 27 | 12"},
    ],
    "pie_chart": [
        # coverage: 1, 2, 2, 3, 4, 4 periods — four-period is the layout that crashed in production
        {"title": "Household expenditure by category in 2022",
         "desc": "Category | Share\nHousing | 34\nTransport | 26\nFood | 22\nLeisure | 18"},
        {"title": "Household spending in 2015 and 2023",
         "desc": "Sector | 2015 | 2023\nHousing | 30 | 38\nTransport | 28 | 22\nFood | 24 | 21\nLeisure | 18 | 19"},
        {"title": "Energy use by sector in 2000 and 2020",
         "desc": "Sector | 2000 | 2020\nIndustry | 40 | 30\nTransport | 30 | 33\nHomes | 20 | 24\nServices | 10 | 13"},
        {"title": "Modes of transport to work in 2000, 2010 and 2020",
         "desc": "Mode | 2000 | 2010 | 2020\nCar | 45 | 42 | 38\nBus | 25 | 24 | 23\nBicycle | 18 | 20 | 22\nWalking | 12 | 14 | 17"},
        {"title": "Household spending across four different years",
         "desc": "Sector | 1990 | 2000 | 2010 | 2020\nHousing | 30 | 32 | 35 | 38\nTransport | 28 | 26 | 23 | 22\nFood | 24 | 23 | 22 | 21\nLeisure | 18 | 19 | 20 | 19"},
        {"title": "Sources of electricity in 1990, 2000, 2010 and 2020",
         "desc": "Source | 1990 | 2000 | 2010 | 2020\nCoal | 40 | 34 | 26 | 18\nGas | 30 | 32 | 33 | 30\nNuclear | 20 | 21 | 22 | 24\nRenewables | 10 | 13 | 19 | 28"},
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
        {"title": "Median weekly earnings by qualification in 2010 and 2020",
         "desc": "Qualification | 2010 | 2020\nDegree | 720 | 865\nDiploma | 560 | 640\nSecondary | 450 | 505\nNo qualification | 360 | 395"},
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
    """1-4 period pie truth: one arc-centre cluster per period, equal radii,
    no overlapping circles (the four-period production crash), slice count
    ~= periods x categories, and every % label inside its own pie."""
    rows, cols = _rows_cols(case["desc"])
    periods = cols - 1
    if _elems(root, "ellipse"):
        return "pie: found <ellipse>"
    arcs = _arc_paths(root)
    if len(arcs) < 2:
        return f"pie: only {len(arcs)} arc <path> elements, expected >=2"
    bad = [(rx, ry) for _, rx, ry, _, _ in arcs if abs(rx - ry) > 0.01]
    if bad:
        return f"pie: non-circular arc rx!=ry {bad[:3]}"
    # cluster arc centres (3px tolerance) — one cluster per pie
    clusters = []  # [cx, cy, max_r]
    for _, rx, _, cx, cy in arcs:
        if cx is None:
            continue
        for cl in clusters:
            if abs(cl[0] - cx) <= 3 and abs(cl[1] - cy) <= 3:
                cl[2] = max(cl[2], rx)
                break
        else:
            clusters.append([cx, cy, rx])
    if len(clusters) != periods:
        return f"pie: {len(clusters)} arc-centre groups, expected {periods} (one per period)"
    radii = {round(cl[2]) for cl in clusters}
    if len(radii) != 1:
        return f"pie: unequal radii across pies {sorted(radii)}"
    expected_slices = periods * rows
    if abs(len(arcs) - expected_slices) > periods:
        return f"pie: {len(arcs)} slice paths, expected ~{expected_slices} (tolerance ±{periods})"
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            x1, y1, r1 = clusters[i]
            x2, y2, r2 = clusters[j]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist < r1 + r2 + 8:
                return f"pie: overlap centres ({x1:.0f},{y1:.0f})&({x2:.0f},{y2:.0f}) dist={dist:.0f} < {r1 + r2 + 8:.0f} (2r+8)"
    # % labels must sit inside their own (nearest) pie — catches labels bleeding
    # onto a neighbouring circle
    for t in _elems(root, "text"):
        txt = (t.text or "").strip()
        if re.fullmatch(r"\d+(\.\d+)?%", txt):
            x = float(t.get("x", -1)); y = float(t.get("y", -1))
            dists = [((x - cl[0]) ** 2 + (y - cl[1]) ** 2) ** 0.5 for cl in clusters]
            k = dists.index(min(dists))
            if dists[k] > clusters[k][2]:
                return f"pie: label '{txt}' at ({x:.0f},{y:.0f}) outside its pie (dist={dists[k]:.0f} > r={clusters[k][2]:.0f})"
    # multi-pie: no pie bottom edge may enter the legend band (the 7px graze
    # that slipped past circle-circle checks). Legend top = min y of legend
    # elements at y>=380; skip when no legend is detectable.
    if periods > 1:
        legend_ys = []
        for el in root.iter():
            if _local(el.tag) in ("rect", "text"):
                try:
                    y = float(el.get("y", "nan"))
                except ValueError:
                    continue
                if y >= 380:
                    legend_ys.append(y)
        if legend_ys:
            legend_top = min(legend_ys)
            for cl in clusters:
                if cl[1] + cl[2] > legend_top:
                    return f"pie: bottom edge {cl[1] + cl[2]:.0f} overlaps legend band top {legend_top:.0f}"
    # four-period only: the four period sub-titles must land on the contract's
    # fixed slots (185,68),(415,68),(185,240),(415,240) within ±5px x / ±3px y.
    # (The nearest-pie matcher was retired: with both rows sharing a cx it
    # provably mis-assigned row2's y=240 sub-title to the row1 circle.)
    # 2/3-period contracts place sub-titles BELOW pies by design — not checked.
    if periods >= 4:
        subtitles = []
        for t in _elems(root, "text"):
            txt = (t.text or "").strip()
            if not re.fullmatch(r"(?:19|20)\d{2}s?", txt):
                continue
            try:
                x = float(t.get("x", "nan")); y = float(t.get("y", "nan"))
            except ValueError:
                continue
            if y != y or y <= 45:  # NaN, or the chart-title band
                continue
            subtitles.append((txt, x, y))
        # existence is mandatory: zero sub-titles → found=0 → fail (no skip escape)
        if len(subtitles) != 4:
            return f"pie: {len(subtitles)} period sub-titles, expected exactly 4"
        slots = [(185, 68), (415, 68), (185, 240), (415, 240)]
        for txt, x, y in subtitles:
            for s in slots:
                if abs(x - s[0]) <= 5 and abs(y - s[1]) <= 3:
                    slots.remove(s)
                    break
            else:
                return f"pie: sub-title '{txt}' at ({x:.0f},{y:.0f}) matches no fixed slot in {{(185,68),(415,68),(185,240),(415,240)}}"
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


# Assertion-strength conservation (2026-07-06 revision): this revision only
# widens "what counts as a legitimately drawn series" (segment-per-segment
# palette <line> chains are valid SVG line charts) and partitions circles into
# chart-area markers vs legend swatches (the geometry contract ITSELF mandates
# a legend at y=360-415 — bounds-checking those circles against the chart area
# punished a required element). No geometric truth was loosened: pie still
# rejects any <ellipse>, bar rects must still touch baseline 340, table still
# needs >= rows*cols text nodes, and line charts still need a real drawn
# series plus every chart-area marker inside x=70..560 / y=40..345.
_SERIES_PALETTE = {"#1a3550", "#c9a84c", "#2d5016", "#6b1a1a", "#7a4b7a"}


def check_line(root, case):
    rows, _ = _rows_cols(case["desc"])
    chart_markers = 0
    for c in _elems(root, "circle"):
        cx = float(c.get("cx", -1)); cy = float(c.get("cy", -1))
        if cy <= 345:
            # chart-area marker: must sit inside the chart area proper
            if not (70 <= cx <= 560 and 40 <= cy <= 345):
                return f"line: chart-area marker ({cx},{cy}) outside x=70..560 y=40..345"
            chart_markers += 1
        elif 360 <= cy <= 415:
            # legend swatch: exempt from chart-area bounds, must stay on canvas
            if not (0 <= cx <= 600 and cy <= 420):
                return f"line: legend circle ({cx},{cy}) outside canvas"
        else:
            return f"line: suspicious circle ({cx},{cy}) between chart area and legend band"
    has_series = bool(_elems(root, "polyline")) or any(
        "L" in (p.get("d") or "").upper() for p in _elems(root, "path"))
    if not has_series:
        # segment-per-segment series: >=2 palette-coloured <line> at width >=2;
        # axes/gridlines (#333 / #e0e0e0, width <=1) never count
        segments = 0
        for ln in _elems(root, "line"):
            stroke = (ln.get("stroke") or "").lower()
            try:
                sw = float(ln.get("stroke-width", 0))
            except ValueError:
                sw = 0
            if sw >= 2 and stroke in _SERIES_PALETTE:
                segments += 1
        has_series = segments >= 2
    if not has_series:
        return "line: no <polyline>, no line-to <path>, and <2 palette-series <line> segments"
    if chart_markers < rows:
        return f"line: {chart_markers} chart-area markers, expected >= {rows}"
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
    # optional argv filter, e.g. `verify_task1_charts.py line_graph` reruns one
    # subtype only (5 calls) without re-spending the other fifteen
    only = set(sys.argv[1:]) & set(CASES) if sys.argv[1:] else set(CASES)
    jobs = [(st, i, c) for st, cases in CASES.items() if st in only for i, c in enumerate(cases)]
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
    print(f"{'subtype':<12} | passed/N | first_failure_reason")
    print("-" * 90)
    frozen = []
    for st in CASES:
        if st not in only:
            continue
        s = summary[st]
        n = len(CASES[st])
        first = s["fails"][0] if s["fails"] else "-"
        print(f"{st:<12} | {s['pass']}/{n}      | {first}")
        if s["pass"] < n:
            frozen.append((st, first))
    # Single-file eyeball index: every generated SVG inlined into one HTML page.
    svgs = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".svg"))
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write("<meta charset='utf-8'><title>Task 1 chart truth-check</title>"
                "<style>body{font-family:Georgia,serif}figure{display:inline-block;"
                "margin:12px;border:1px solid #ccc;padding:8px}figcaption{font-size:13px;"
                "text-align:center}svg{width:480px;height:auto}</style>\n")
        for name in svgs:
            with open(os.path.join(OUT_DIR, name)) as s:
                f.write(f"<figure>{s.read()}<figcaption>{name}</figcaption></figure>\n")

    print()
    print(f"PNG rendering: {'yes' if HAVE_PNG else 'NO — libcairo absent; geometry-only asserts, raw .svg written to scripts/out/ for eyeballing'}")
    print(f"Eyeball index: scripts/out/index.html ({len(svgs)} SVGs embedded)")
    if not frozen:
        print("SERVED 4/4 at N/N: YES")
    else:
        detail = ", ".join(f"{st}: {r}" for st, r in frozen)
        print(f"SERVED 4/4 at N/N: NO — frozen: {[st for st, _ in frozen]}, first_failure: {detail}")
    return 0 if not frozen else 1


if __name__ == "__main__":
    sys.exit(main())

"""Download woff2 font files for the dashboard and write fonts.css.

Fonts are self-hosted (the user's Brave mobile blocks Google Fonts, which
made Material Symbols render as literal text). Files land in
app/web/static/fonts/ and are served by the bot itself.

Usage: python scripts/fetch_fonts.py
"""
import re
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
OUT = Path(__file__).resolve().parent.parent / "app" / "web" / "static" / "fonts"

# (family, css2 query, weights wanted) — latin subset only.
FAMILIES = [
    ("Inter", "family=Inter:wght@400;500;600;700", [400, 500, 600, 700]),
    ("JetBrains Mono", "family=JetBrains+Mono:wght@400;500;600;700", [400, 500, 600, 700]),
    ("Space Grotesk", "family=Space+Grotesk:wght@500;600;700", [500, 600, 700]),
]
# Material Symbols Outlined variable font (wght + FILL axes, as the design uses).
SYMBOLS_QUERY = "family=Material+Symbols+Outlined:wght,FILL@100..700,0..1"

ICON_NAMES = [
    "candlestick_chart", "sensors", "check_circle", "power_settings_new",
    "pause_circle", "arrow_drop_up", "arrow_drop_down", "trending_up",
    "trending_down", "verified_user", "event_upcoming", "history",
    "terminal", "monitoring", "analytics", "rule", "event_note", "warning",
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def latin_blocks(css: str) -> list[tuple[int, str]]:
    """Yield (weight, woff2 url) for the /* latin */ @font-face blocks."""
    out = []
    for m in re.finditer(r"/\* (\w[\w-]*) \*/\s*@font-face\s*{([^}]*)}", css):
        subset, body = m.group(1), m.group(2)
        if subset != "latin":
            continue
        weight = int(re.search(r"font-weight:\s*(\d+)", body).group(1))
        url = re.search(r"url\((https://[^)]+\.woff2)\)", body).group(1)
        out.append((weight, url))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    css_lines = []
    for family, query, weights in FAMILIES:
        css = fetch(f"https://fonts.googleapis.com/css2?{query}&display=swap").decode()
        for weight, url in latin_blocks(css):
            if weight not in weights:
                continue
            fname = f"{family.replace(' ', '')}-{weight}.woff2"
            data = fetch(url)
            (OUT / fname).write_bytes(data)
            print(f"{fname}: {len(data)} bytes")
            css_lines.append(
                "@font-face{font-family:'%s';font-style:normal;font-weight:%d;"
                "font-display:swap;src:url(/static/fonts/%s) format('woff2');}"
                % (family, weight, fname)
            )
    # Material Symbols variable font: take the latin block (unicode-range
    # U+0000-00FF...) — the icon ligatures live in the base font.
    css = fetch(f"https://fonts.googleapis.com/css2?{SYMBOLS_QUERY}&display=swap").decode()
    blocks = re.findall(r"@font-face\s*{([^}]*)}", css)
    picked = None
    for body in blocks:
        if "U+0000-00FF" in body:
            picked = body
            break
    if picked is None:
        picked = blocks[-1]
    url = re.search(r"url\((https://[^)]+\.woff2)\)", picked).group(1)
    data = fetch(url)
    (OUT / "MaterialSymbolsOutlined.woff2").write_bytes(data)
    print(f"MaterialSymbolsOutlined.woff2: {len(data)} bytes")
    css_lines.append(
        "@font-face{font-family:'Material Symbols Outlined';font-style:normal;"
        "font-weight:100 700;font-display:swap;"
        "src:url(/static/fonts/MaterialSymbolsOutlined.woff2) format('woff2');}"
    )
    (OUT / "fonts.css").write_text("\n".join(css_lines) + "\n", encoding="utf-8")
    print("fonts.css written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

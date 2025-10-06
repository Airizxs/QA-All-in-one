import os
import json
import sys
import argparse
import re
import threading
import time
import itertools
import subprocess
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timezone

# Silence urllib3's LibreSSL warning early (before importing requests/urllib3)
import warnings
# Match the specific message emitted by urllib3 on LibreSSL builds
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL.*")

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

try:  # Optional rich rendering for professional output
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError:  # pragma: no cover - optional dependency
    Console = None
    Table = None
    Panel = None
    box = None

# Import your individual check functions
from .checks.title_meta import check_title_and_meta
from .checks.headings import check_headings
from .checks.schema import check_schema
from .checks.mobile import check_mobile_responsiveness
from .checks.robots_sitemaps import check_robots_and_sitemaps
from .checks.internal_links import check_internal_links
from .checks.images import check_images
from .checks.canonical import check_canonical_and_hreflang
from .checks.indexability import check_indexability
from .checks.faq import check_faq
from .checks.locations import check_locations
from .checks.hero import check_hero_image
from .utils.fetch import fetch_html, build_session, DEFAULT_HEADERS


_RICH_CONSOLE: Optional[Console] = None
if Console is not None and os.environ.get("NO_COLOR") is None:
    _RICH_CONSOLE = Console()

_STATUS_STYLE_MAP = {
    'pass': 'bold green',
    'ok': 'bold green',
    'found': 'bold green',
    'present': 'bold green',
    'yes': 'bold green',
    'success': 'bold green',
    'warning': 'bold yellow',
    'warn': 'bold yellow',
    'fail': 'bold red',
    'missing': 'bold red',
    'no': 'bold red',
    'error': 'bold red',
}

_PASS_STATUSES = {'pass', 'ok', 'found', 'present', 'yes', 'success'}
 

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SEO checks on a URL")
    parser.add_argument("urls", nargs="*", help="One or more target URLs to audit")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout in seconds (default: 20)")
    parser.add_argument("--use-scraperapi", action="store_true", help="Use ScraperAPI if SCRAPERAPI_KEY is set")
    parser.add_argument("--max-links", type=int, default=25, help="Max internal links to verify (default: 25)")
    parser.add_argument("--output-json", type=str, help="Path to write JSON results")
    parser.add_argument("--output-csv", type=str, help="Path to write CSV summary")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error console output")
    parser.add_argument(
        "--format",
        choices=["table", "json", "both", "text", "table+text", "text+json"],
        default="table",
        help="Console output format: table, text, json, or combinations (default: table)",
    )
    parser.add_argument(
        "--threshold", type=float, default=80.0,
        help="QA pass threshold as percentage (default: 80)")
    parser.add_argument(
        "--url-file", type=str, help="Optional file with URLs (one per line) to include"
    )
    parser.add_argument(
        "--history-file", type=str, default=".seo_checker_history.jsonl",
        help="Path to history file (JSON Lines). Default: .seo_checker_history.jsonl"
    )
    parser.add_argument(
        "--no-history", action="store_true",
        help="Disable appending results to the history file"
    )
    parser.add_argument(
        "--show-history", nargs="?", const=20, type=int,
        help="Show the last N history entries (default 20) and exit"
    )
    parser.add_argument(
        "--keyword", type=str,
        help="Optional target keyword to verify in title/description"
    )
    parser.add_argument(
        "--child-url", type=str,
        help="Optional child page URL to compare format/template against the main URL"
    )
    # Spelling and AI options removed
    # Firewall/proxy options
    parser.add_argument(
        "--proxy", type=str,
        help="HTTP(S) proxy URL to use for all requests (overrides env HTTP(S)_PROXY)"
    )
    parser.add_argument(
        "--ca-bundle", type=str,
        help="Path to custom CA bundle file for TLS verification (sets REQUESTS_CA_BUNDLE)"
    )
    parser.add_argument(
        "--insecure", action="store_true",
        help="Disable TLS verification (not recommended)."
    )
    parser.add_argument(
        "--capture-screenshots",
        action="store_true",
        help="Capture Puppeteer screenshots (requires Node.js and tools/capture_sections.js).",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default="logs/screenshots",
        help="Directory for Puppeteer screenshots (default: logs/screenshots)",
    )
    parser.add_argument(
        "--puppeteer-script",
        type=str,
        default="tools/capture_sections.js",
        help="Override path to Puppeteer capture script.",
    )
    parser.add_argument(
        "--screenshot-clip",
        type=int,
        default=1000,
        help="Viewport height (px) captured for responsiveness composites.",
    )
    return parser.parse_args(argv)

def run_all_checks(url: str, *, timeout: int = 20, use_scraperapi: bool = False, max_links: int = 25, quiet: bool = False, keyword: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs all defined SEO checks on a given URL.
    
    Args:
        url (str): The URL of the website to check.
        
    Returns:
        dict: A dictionary containing the results of all checks.
    """
    if not quiet:
        print(f"Starting SEO checks for: {url}")
    results = {}

    spinner = Spinner("Fetching HTML", enabled=_tty_color_enabled() and not quiet)
    spinner.start()
    html = fetch_html(url, timeout=timeout, use_scraperapi=use_scraperapi)
    spinner.stop("Fetched HTML" if html else "Fetch failed")
    if not html:
        return {"error": "Failed to access the URL. Consider setting SCRAPERAPI_KEY for tougher sites."}

    soup = BeautifulSoup(html, 'html.parser')
    # Build heuristic style signature for later parent/child comparison
    try:
        style_sig = _build_style_signature(soup, url)
    except Exception:
        style_sig = {}

    def _spin_step(label: str, fn):
        sp = Spinner(f"{label}", enabled=_tty_color_enabled() and not quiet)
        sp.start()
        try:
            out = fn()
            sp.stop(f"{label} done")
            return out
        except Exception as e:
            sp.stop(f"{label} failed")
            return {"status": "error", "message": f"{label} error: {e}"}

    # Run each check and store the results
    results['title_meta'] = _spin_step("Title & Meta", lambda: check_title_and_meta(soup, keyword=keyword))
    results['headings'] = _spin_step("Headings", lambda: check_headings(soup))
    results['schema'] = _spin_step("Schema", lambda: check_schema(soup))
    results['mobile_responsiveness'] = _spin_step("Mobile", lambda: check_mobile_responsiveness(url))
    # New checks
    results['robots_sitemaps'] = _spin_step("Robots & Sitemaps", lambda: check_robots_and_sitemaps(url, timeout=timeout, use_scraperapi=use_scraperapi))
    results['internal_links'] = _spin_step("Internal Links", lambda: check_internal_links(html, url, timeout=timeout, max_links=max_links))
    results['images'] = _spin_step("Images", lambda: check_images(soup, url, fetch_sizes=True, max_fetch=20, timeout=timeout))
    results['canonical_hreflang'] = _spin_step("Canonical & Hreflang", lambda: check_canonical_and_hreflang(soup, url))
    results['indexability'] = _spin_step("Indexability", lambda: check_indexability(url, soup))
    results['faq'] = _spin_step("FAQ", lambda: check_faq(soup))
    results['locations'] = _spin_step("Locations", lambda: check_locations(soup))
    results['hero_image'] = _spin_step("Hero Image", lambda: check_hero_image(soup))
    # Layout check removed
    # Spelling and AI review removed

    # Cross-check: author meta vs schema authors
    def _author_match():
        tm_author = (results.get('title_meta', {}).get('author', {}) or {}).get('content')
        sc_authors = [a.strip() for a in (results.get('schema', {}).get('authors') or []) if isinstance(a, str)]
        if tm_author and sc_authors:
            if any(tm_author.lower() == a.lower() for a in sc_authors):
                results['title_meta']['author']['status'] = 'ok'
                results['title_meta']['author']['message'] = 'Author matches schema.'
            else:
                results['title_meta']['author']['status'] = 'warning'
                results['title_meta']['author']['message'] = 'Author meta does not match schema author(s).'
        return True
    _spin_step("Author Match", _author_match)

    # Attach style signature for comparison output
    results['_style_signature'] = style_sig
    results['_fetched_at'] = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')

    if not quiet:
        print(colorize("All checks completed.", "info"))
    return results

def _truncate(value: Any, limit: int = 80) -> str:
    s = ", ".join(value) if isinstance(value, list) else str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def _tty_color_enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def colorize(text: str, status: Optional[str] = None) -> str:
    if not _tty_color_enabled():
        return text
    s = (status or "").lower()
    # Map statuses and common words to colors
    if s in ("pass", "ok", "found", "yes", "present", "info", "success"):
        color = "32"  # green
    elif s in ("warning", "warn"):
        color = "33"  # yellow
    elif s in ("fail", "missing", "no", "error"):
        color = "31"  # red
    else:
        # fallback neutral (cyan for info text if explicitly requested)
        color = "36" if s == "info" else "0"
    return f"\x1b[{color}m{text}\x1b[0m" if color != "0" else text

def _color_for_cell(header: str, value: str) -> Optional[str]:
    key = header.strip().lower()
    val = (value or "").strip().lower()
    if key in ("status", "present"):
        # Normalize booleans
        if val in ("true", "yes"):
            return "pass"
        if val in ("false", "no"):
            return "fail"
        return val  # already a status string
    return None


def _rich_enabled() -> bool:
    return _RICH_CONSOLE is not None and getattr(_RICH_CONSOLE, "is_terminal", False)


def _pad_visible(s: str, width: int) -> str:
    # Pad based on visible length (exclude ANSI sequences)
    length = len(_strip_ansi(s))
    if length < width:
        return s + (" " * (width - length))
    return s

def _print_table(title: str, headers: List[str], rows: List[List[Any]]):
    if _rich_enabled():
        table = Table(title=title, box=box.ROUNDED if box else None, header_style="bold cyan")
        for header in headers:
            table.add_column(header)

        for row in rows or []:
            cells: List[str] = []
            for idx in range(len(headers)):
                value = "" if row is None or idx >= len(row) or row[idx] is None else str(row[idx])
                hint = _color_for_cell(headers[idx], value)
                style = _STATUS_STYLE_MAP.get((hint or "").lower()) if hint else None
                if style:
                    cells.append(f"[{style}]{value}[/{style}]")
                else:
                    cells.append(value)
            table.add_row(*cells)

        if not rows:
            table.add_row(*(["-"] * len(headers)))

        _RICH_CONSOLE.print(table)
        return

    print(f"\n== {title} ==")
    cols = len(headers)
    widths = [len(h) for h in headers]
    raw_rows: List[List[str]] = []
    for row in rows:
        raw = ["" if i >= len(row) or row[i] is None else str(row[i]) for i in range(cols)]
        raw_rows.append(raw)
        for i in range(cols):
            vis_len = len(_strip_ansi(raw[i]))
            if vis_len > widths[i]:
                widths[i] = vis_len
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(cols)) + "-|"
    print(header_line)
    print(sep_line)
    for raw in raw_rows:
        colored_cells: List[str] = []
        for i in range(cols):
            header = headers[i]
            val = raw[i]
            status_hint = _color_for_cell(header, val)
            cell = colorize(val, status_hint) if status_hint else val
            colored_cells.append(_pad_visible(cell, widths[i]))
        print("| " + " | ".join(colored_cells) + " |")


def _run_puppeteer_capture(script_path: str, url: str, output_dir: str, clip_height: int = 1000) -> Tuple[bool, str]:
    script_full = os.path.abspath(script_path)
    if not os.path.exists(script_full):
        return False, f"Capture script not found at {script_full}"

    output_full = os.path.abspath(output_dir)
    cmd = [
        "node",
        script_full,
        url,
        "--output",
        output_full,
        "--clip",
        str(max(400, clip_height)),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
    except FileNotFoundError:
        return False, "Node.js runtime not found; install Node to enable screenshots."
    except subprocess.TimeoutExpired:
        return False, "Puppeteer capture timed out."

    if completed.returncode != 0:
        combined = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return False, combined

    message = completed.stdout.strip() or f"Screenshots saved to {output_full}"
    return True, message


def _collect_recommendations(result: Dict[str, Any]) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    seen: Set[tuple] = set()

    def needs_attention(status: Optional[str]) -> bool:
        if not status:
            return False
        return str(status).lower() not in _PASS_STATUSES

    def add(area: str, status: Optional[str], message: Optional[str]):
        if not message:
            return
        key = (area, message.strip())
        if key in seen:
            return
        seen.add(key)
        recs.append({
            "area": area,
            "status": (status or "").upper(),
            "message": message.strip(),
        })

    summary = result.get('_score_summary', {})
    if summary and summary.get('result') == 'FAIL':
        add(
            "Overall Score",
            "fail",
            f"Score {summary.get('percent', 0)}% is below the {summary.get('threshold', 80)}% threshold.",
        )

    if result.get('error'):
        add("Fetch", "error", result.get('error'))

    tm = result.get('title_meta', {})
    title = tm.get('title', {})
    if needs_attention(title.get('status')):
        add("Title Tag", title.get('status'), title.get('message') or "Refine the <title> tag for clarity and keyword focus.")
    meta_desc = tm.get('meta_description', {})
    if needs_attention(meta_desc.get('status')):
        add("Meta Description", meta_desc.get('status'), meta_desc.get('message') or "Improve the meta description length and relevance.")
    author = tm.get('author', {})
    if needs_attention(author.get('status')):
        add("Author Meta", author.get('status'), author.get('message') or "Populate author metadata to align with schema.")

    headings = result.get('headings', {})
    if needs_attention(headings.get('h1_status')):
        add("H1 Heading", headings.get('h1_status'), "Ensure a single descriptive H1 heading exists.")
    if needs_attention(headings.get('h_hierarchy')):
        hierarchy_msg = headings.get('h_hierarchy_message') or "Fix heading levels to avoid skipped hierarchy (e.g., H2 → H4)."
        add("Heading Hierarchy", headings.get('h_hierarchy'), hierarchy_msg)

    schema = result.get('schema', {})
    if not schema.get('schema_found'):
        add("Schema Markup", "fail", "Add JSON-LD schema to qualify for rich results.")

    mobile = result.get('mobile_responsiveness', {})
    if needs_attention(mobile.get('status')):
        add("Mobile", mobile.get('status'), mobile.get('message') or "Review viewport settings and responsive breakpoints.")

    indexability = result.get('indexability', {})
    if needs_attention(indexability.get('status')):
        add("Indexability", indexability.get('status'), indexability.get('message') or "Check robots directives and headers to ensure the page is indexable.")

    robots_info = result.get('robots_sitemaps', {})
    robots = robots_info.get('robots', {})
    if not robots.get('present'):
        add("Robots.txt", "fail", "Publish a robots.txt file with relevant directives.")
    sitemaps = robots_info.get('sitemaps', {})
    if needs_attention(sitemaps.get('status')):
        add("Sitemaps", sitemaps.get('status'), sitemaps.get('message') or "Validate sitemap URLs and ensure they are accessible.")

    internal_links = result.get('internal_links', {})
    if needs_attention(internal_links.get('status')):
        add("Internal Links", internal_links.get('status'), internal_links.get('message') or "Repair broken internal links and improve contextual linking.")

    images = result.get('images', {})
    if needs_attention(images.get('status')):
        add("Images", images.get('status'), images.get('message') or "Audit image alt text and file sizes.")

    faq = result.get('faq', {})
    if needs_attention(faq.get('status')):
        add("FAQ", faq.get('status'), faq.get('message') or "Ensure FAQ items use H3 headings and include FAQPage schema.")

    hero = result.get('hero_image', {})
    if needs_attention(hero.get('status')):
        add("Hero Image", hero.get('status'), hero.get('message') or "Supply explicit width/height for the hero image.")

    locations = result.get('locations', {})
    if needs_attention(locations.get('status')):
        add("Locations", locations.get('status'), locations.get('message') or "Add structured address details for each location.")

    return recs


def _print_recommendations(result: Dict[str, Any]):
    recs = _collect_recommendations(result)
    if not recs:
        message = "No outstanding action items – great job!"
        if _rich_enabled():
            _RICH_CONSOLE.print(Panel.fit(f"[bold green]{message}", title="Recommendations", border_style="green"))
        else:
            print("\nRecommendations:\n- " + message)
        return

    if _rich_enabled():
        table = Table(title="Recommendations", box=box.ROUNDED if box else None, header_style="bold magenta")
        table.add_column("Area", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Recommended Action", overflow="fold")
        for rec in recs:
            status_lower = rec['status'].lower()
            style = _STATUS_STYLE_MAP.get(status_lower)
            status_cell = rec['status']
            if style:
                status_cell = f"[{style}]{rec['status']}[/{style}]"
            table.add_row(rec['area'], status_cell, rec['message'])
        _RICH_CONSOLE.print(table)
    else:
        print("\nRecommendations:")
        for rec in recs:
            print(f"- [{rec['status']}] {rec['area']}: {rec['message']}")

def _parse_style_attr(style: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not style:
        return out
    try:
        for part in str(style).split(';'):
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip().lower()
                v = v.strip().lower()
                if k:
                    out[k] = v
    except Exception:
        pass
    return out

def _collect_css_text(soup: BeautifulSoup, base_url: str, *, max_files: int = 5, max_bytes: int = 200_000, timeout: int = 10) -> str:
    chunks: List[str] = []
    # Inline <style>
    for st in soup.find_all('style'):
        txt = st.string or st.get_text() or ''
        if txt:
            chunks.append(txt)
    # Linked stylesheets
    links: List[str] = []
    for ln in soup.find_all('link'):
        rel = (ln.get('rel') or [])
        rels = {r.lower() for r in (rel if isinstance(rel, list) else [rel]) if r}
        as_attr = (ln.get('as') or '').lower()
        if 'stylesheet' in rels or ('preload' in rels and as_attr == 'style'):
            href = ln.get('href')
            if href:
                links.append(urljoin(base_url, href))
    session = build_session(DEFAULT_HEADERS)
    seen = set()
    for href in links:
        if href in seen:
            continue
        seen.add(href)
        if len(seen) > max_files:
            break
        try:
            r = session.get(href, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            txt = r.text
            if len(txt) > max_bytes:
                txt = txt[:max_bytes]
            chunks.append(txt)
        except Exception:
            continue
    return "\n".join(chunks)


def _extract_fonts_colors_from_css(css_text: str) -> Dict[str, List[str]]:
    fonts: List[str] = []
    tcolors: List[str] = []
    bfonts: List[str] = []
    import re as _re
    # Common selectors for text
    for sel in (r'body', r'p', r'a', r'h1', r'h2', r'h3', r'h4', r'h5', r'h6'):
        m = _re.search(rf"{sel}\s*\{{[^}}]*?font-family\s*:\s*([^;}}]+)", css_text, _re.I | _re.S)
        if m:
            fonts.append(m.group(1).strip().lower())
        m2 = _re.search(rf"{sel}\s*\{{[^}}]*?color\s*:\s*([^;}}]+)", css_text, _re.I | _re.S)
        if m2:
            tcolors.append(m2.group(1).strip().lower())
    # Buttons (generic .btn patterns and button tag)
    for sel in (r'button', r'\.btn[\w-]*', r'a\.btn[\w-]*'):
        m = _re.search(rf"{sel}\s*\{{[^}}]*?font-family\s*:\s*([^;}}]+)", css_text, _re.I | _re.S)
        if m:
            bfonts.append(m.group(1).strip().lower())
    return {
        'text_fonts': sorted(list({f for f in fonts if f})),
        'text_colors': sorted(list({c for c in tcolors if c})),
        'button_fonts': sorted(list({bf for bf in bfonts if bf})),
    }


def _build_style_signature(soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
    sig: Dict[str, Any] = {}
    # Bold/italic detection via tags and inline styles
    bold_tags = soup.find_all(['strong', 'b'])
    italic_tags = soup.find_all(['em', 'i'])
    styled_bold = 0
    styled_italic = 0
    for el in soup.find_all(True):
        st = _parse_style_attr(el.get('style'))
        if not st:
            continue
        fw = st.get('font-weight')
        fs = st.get('font-style')
        try:
            if fw and (fw in ('bold', 'bolder') or (fw.isdigit() and int(fw) >= 600)):
                styled_bold += 1
        except Exception:
            pass
        if fs and fs == 'italic':
            styled_italic += 1
    sig['bold_count'] = len(bold_tags) + styled_bold
    sig['italic_count'] = len(italic_tags) + styled_italic

    # Text colors (inline only, heuristic)
    color_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'li'])
    text_colors = set()
    text_fonts = set()
    for el in color_tags[:500]:
        st = _parse_style_attr(el.get('style'))
        col = st.get('color')
        if col:
            text_colors.add(col)
        ff = st.get('font-family')
        if ff:
            text_fonts.add(ff)
    sig['text_colors'] = sorted(text_colors)
    sig['text_fonts'] = sorted(text_fonts)

    # Buttons: tag, classes, inline colors
    buttons = []
    button_fonts = set()
    for el in soup.find_all(['button', 'a']):
        role = (el.get('role') or '').lower()
        classes = ' '.join((el.get('class') or [])[:3]).strip().lower()
        if el.name == 'button' or role == 'button' or ('btn' in classes) or ('button' in classes):
            st = _parse_style_attr(el.get('style'))
            buttons.append({
                'classes': classes,
                'bg': st.get('background-color'),
                'color': st.get('color'),
            })
            ff = st.get('font-family')
            if ff:
                button_fonts.add(ff)
        if len(buttons) >= 50:
            break
    sig['button_bg_colors'] = sorted({b['bg'] for b in buttons if b.get('bg')})
    sig['button_text_colors'] = sorted({b['color'] for b in buttons if b.get('color')})
    sig['button_fonts'] = sorted(button_fonts)

    # FAQ style markers (class names containing 'faq')
    # (faq classes removed)
    # If inline signals are sparse, try to infer from CSS files
    if (not sig['text_colors'] or not sig.get('text_fonts') or not sig.get('button_fonts')):
        try:
            css_txt = _collect_css_text(soup, base_url)
            inferred = _extract_fonts_colors_from_css(css_txt)
            if (not sig.get('text_fonts')) and inferred.get('text_fonts'):
                sig['text_fonts'] = inferred['text_fonts']
            if (not sig.get('text_colors')) and inferred.get('text_colors'):
                sig['text_colors'] = inferred['text_colors']
            if (not sig.get('button_fonts')) and inferred.get('button_fonts'):
                sig['button_fonts'] = inferred['button_fonts']
        except Exception:
            pass
    return sig

def _format_signature(res: Dict[str, Any]) -> Dict[str, Any]:
    sig: Dict[str, Any] = {}
    # Mobile viewport
    mv = res.get('mobile_responsiveness', {})
    sig['viewport'] = bool((mv.get('viewport_meta_tag') or {}).get('found'))
    # Headings structure
    hd = res.get('headings', {})
    sig['heading_levels'] = list(hd.get('h_tags_found') or [])
    # Schema types
    sc = res.get('schema', {})
    types = [str(t).lower() for t in (sc.get('types') or [])]
    sig['schema_types'] = sorted(set(types))
    # FAQ presence
    fq = res.get('faq', {})
    sig['faq_h3'] = int(fq.get('h3_count') or 0) > 0
    sig['faq_schema'] = bool(sc.get('faqpage_found'))
    # Canonical/hreflang
    ch = res.get('canonical_hreflang', {})
    sig['canonical_present'] = bool((ch.get('canonical') or {}).get('status') in ('pass','ok','found'))
    sig['hreflang_count'] = len((ch.get('hreflang') or {}).get('entries') or [])
    # Locations present
    lo = res.get('locations', {})
    sig['has_location'] = int(lo.get('count') or 0) > 0
    # Style signature (heuristic)
    st = res.get('_style_signature', {})
    sig['bold_count'] = st.get('bold_count')
    sig['italic_count'] = st.get('italic_count')
    sig['text_colors'] = st.get('text_colors') or []
    sig['text_fonts'] = st.get('text_fonts') or []
    sig['button_bg_colors'] = st.get('button_bg_colors') or []
    sig['button_text_colors'] = st.get('button_text_colors') or []
    sig['button_fonts'] = st.get('button_fonts') or []
    # faq_classes removed
    return sig

def _print_format_comparison(url_a: str, res_a: Dict[str, Any], url_b: str, res_b: Dict[str, Any]):
    sig_a = _format_signature(res_a)
    sig_b = _format_signature(res_b)
    rows: List[List[Any]] = []

    def as_display(val: Any) -> str:
        if isinstance(val, list):
            return ','.join([str(x) for x in val]) or '-'
        if val in (None, ''):
            return '-'
        return str(val)

    def as_set(val: Any) -> set:
        if isinstance(val, list):
            return {str(x).strip() for x in val if str(x).strip()}
        if isinstance(val, str):
            return {x.strip() for x in val.split(',') if x.strip()}
        return {str(val)}

    def add_row(key: str, label: str, val_a: Any, val_b: Any):
        # Normalize comparisons for collections to ignore order/duplicates
        if key in (
            'schema_types', 'text_colors', 'button_bg_colors', 'button_text_colors'
        ):
            match = (as_set(val_a) == as_set(val_b))
        else:
            match = (val_a == val_b)
        rows.append([
            label,
            as_display(val_a)[:80],
            as_display(val_b)[:80],
            'pass' if match else 'fail'
        ])

    add_row('viewport', 'Viewport Meta', sig_a['viewport'], sig_b['viewport'])
    add_row('schema_types', 'Schema Types', sig_a['schema_types'], sig_b['schema_types'])
    add_row('faq_h3', 'FAQ H3 Present', sig_a['faq_h3'], sig_b['faq_h3'])
    add_row('faq_schema', 'FAQPage Schema', sig_a['faq_schema'], sig_b['faq_schema'])
    add_row('canonical_present', 'Canonical Present', sig_a['canonical_present'], sig_b['canonical_present'])
    add_row('has_location', 'Location Present', sig_a['has_location'], sig_b['has_location'])
    add_row('text_colors', 'Text Colors', sig_a['text_colors'], sig_b['text_colors'])
    add_row('text_fonts', 'Text Fonts', sig_a['text_fonts'], sig_b['text_fonts'])
    add_row('button_bg_colors', 'Button BG Colors', sig_a['button_bg_colors'], sig_b['button_bg_colors'])
    add_row('button_text_colors', 'Button Text Colors', sig_a['button_text_colors'], sig_b['button_text_colors'])
    add_row('button_fonts', 'Button Fonts', sig_a['button_fonts'], sig_b['button_fonts'])

    _print_table(
        "Format Consistency (Parent vs Child)",
        ["Item", "Parent", "Child", "Match"],
        rows,
    )

class Spinner:
    def __init__(self, message: str, enabled: bool = True):
        self.message = message
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])  # Braille spinner

    def start(self):
        if not self.enabled:
            return
        def run():
            while not self._stop.is_set():
                frame = next(self._frames)
                sys.stdout.write(f"\r{frame} {self.message}    ")
                sys.stdout.flush()
                time.sleep(0.08)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self, final_message: Optional[str] = None):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        msg = final_message or self.message
        sys.stdout.write(f"\r✔ {msg}        \n")
        sys.stdout.flush()

def _status_to_percent(status: Optional[str]) -> float:
    if not status:
        return 0.0
    s = str(status).lower()
    if s in ("pass", "ok", "found", "yes"):
        return 100.0
    if s in ("warning",):
        return 50.0
    return 0.0

def _points_from_status(status: Optional[str]) -> float:
    if not status:
        return 0.0
    s = str(status).lower()
    if s in ("pass", "ok", "found"):
        return 1.0
    if s in ("warning",):
        return 0.5
    return 0.0

def _has_nonpass_status(obj: Any) -> bool:
    # Recursively scan for any status == 'fail' or 'warning'
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'status' and isinstance(v, str) and v.lower() in ('fail', 'warning'):
                return True
            if _has_nonpass_status(v):
                return True
        return False
    if isinstance(obj, list):
        return any(_has_nonpass_status(it) for it in obj)
    return False


def compute_score(all_results: Dict[str, Any], *, threshold: float = 80.0) -> Dict[str, Any]:
    score = 0.0
    max_points = 0.0
    # Title & Meta (3)
    tm = all_results.get('title_meta', {})
    score += _points_from_status(tm.get('title', {}).get('status'))
    score += _points_from_status(tm.get('meta_description', {}).get('status'))
    score += _points_from_status(tm.get('author', {}).get('status'))
    max_points += 3
    # Headings (2)
    hd = all_results.get('headings', {})
    score += _points_from_status(hd.get('h1_status'))
    score += _points_from_status(hd.get('h_hierarchy'))
    max_points += 2
    # Schema (1)
    sc = all_results.get('schema', {})
    score += 1.0 if sc.get('schema_found') else 0.0
    max_points += 1
    # Mobile (1)
    mb = all_results.get('mobile_responsiveness', {})
    score += _points_from_status(mb.get('status'))
    max_points += 1
    # Indexability (1)
    ix_all = all_results.get('indexability', {})
    score += _points_from_status(ix_all.get('status'))
    max_points += 1
    # Robots (1)
    rs = all_results.get('robots_sitemaps', {})
    score += 1.0 if rs.get('robots', {}).get('present') else 0.0
    max_points += 1
    # Sitemaps (1)
    score += _points_from_status(rs.get('sitemaps', {}).get('status'))
    max_points += 1
    # Internal links (1)
    il = all_results.get('internal_links', {})
    score += _points_from_status(il.get('status'))
    max_points += 1
    # Images (1)
    im = all_results.get('images', {})
    score += _points_from_status(im.get('status'))
    max_points += 1
    # Canonical (1) + Hreflang (1)
    ch = all_results.get('canonical_hreflang', {})
    score += _points_from_status(ch.get('canonical', {}).get('status'))
    score += _points_from_status(ch.get('hreflang', {}).get('status'))
    max_points += 2
    # Locations (1)
    lo = all_results.get('locations', {})
    score += _points_from_status(lo.get('status'))
    max_points += 1
    # Hero image (1)
    hero = all_results.get('hero_image', {})
    score += _points_from_status(hero.get('status'))
    max_points += 1
    percent = (score / max_points * 100.0) if max_points else 0.0
    # Strict mode: any warning/fail anywhere forces FAIL
    strict_fail = _has_nonpass_status(all_results)
    qa_result = "FAIL" if strict_fail else ("PASS" if percent >= threshold else "FAIL")
    return {
        'score': round(score, 2),
        'max': int(max_points),
        'percent': round(percent, 1),
        'threshold': threshold,
        'result': qa_result,
    }

def compute_section_scores(res: Dict[str, Any]) -> Dict[str, float]:
    section_scores: Dict[str, float] = {}
    # Title & Meta: average of three
    tm = res.get('title_meta', {})
    title_pct = _status_to_percent(tm.get('title', {}).get('status'))
    meta_pct = _status_to_percent(tm.get('meta_description', {}).get('status'))
    author_pct = _status_to_percent(tm.get('author', {}).get('status'))
    section_scores['title_meta'] = (title_pct + meta_pct + author_pct) / 3.0

    # Headings: H1 + hierarchy
    hd = res.get('headings', {})
    h1_pct = _status_to_percent(hd.get('h1_status'))
    hier_pct = _status_to_percent(hd.get('h_hierarchy'))
    section_scores['headings'] = (h1_pct + hier_pct) / 2.0

    # Schema
    sc = res.get('schema', {})
    section_scores['schema'] = 100.0 if sc.get('schema_found') else 0.0

    # Mobile
    mb = res.get('mobile_responsiveness', {})
    section_scores['mobile'] = _status_to_percent(mb.get('status'))

    # Robots
    rs = res.get('robots_sitemaps', {})
    section_scores['robots'] = 100.0 if rs.get('robots', {}).get('present') else 0.0
    # Sitemaps
    section_scores['sitemaps'] = _status_to_percent(rs.get('sitemaps', {}).get('status'))

    # Internal links: ratio based
    il = res.get('internal_links', {})
    checked = max(1, int(il.get('checked') or 0))
    broken_count = len(il.get('broken') or [])
    if (il.get('checked') or 0) > 0:
        section_scores['internal_links'] = max(0.0, 100.0 * (1.0 - broken_count / float(checked)))
    else:
        section_scores['internal_links'] = 0.0

    # Images: ratio based with poor-alt penalty
    im = res.get('images', {})
    total_imgs = int(im.get('total_images') or 0)
    missing_alt = len(im.get('missing_alt') or [])
    if total_imgs == 0:
        total_found = int(im.get('total_found') or 0)
        hidden = int(im.get('skipped_hidden') or 0)
        if total_found > 0 and hidden == total_found:
            section_scores['images'] = 100.0
        else:
            section_scores['images'] = 0.0
    else:
        section_scores['images'] = max(0.0, 100.0 * (1.0 - missing_alt / float(total_imgs)))

    # Indexability
    ix = res.get('indexability', {})
    section_scores['indexability'] = _status_to_percent(ix.get('status'))

    # Canonical & hreflang
    ch = res.get('canonical_hreflang', {})
    section_scores['canonical'] = _status_to_percent(ch.get('canonical', {}).get('status'))
    section_scores['hreflang'] = _status_to_percent(ch.get('hreflang', {}).get('status'))

    # Locations
    lo = res.get('locations', {})
    section_scores['locations'] = _status_to_percent(lo.get('status'))

    hero = res.get('hero_image', {})
    section_scores['hero_image'] = _status_to_percent(hero.get('status'))

    return section_scores


def _status_text(status: Optional[str]) -> str:
    return str(status).upper() if status else "N/A"


def _clean_copy_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(v) for v in value if v is not None)
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def _build_summary_paragraph(results: Dict[str, Any], url: str) -> str:
    fragments: List[str] = []

    fragments.append(f"SEO audit for {url} completed.")

    tm = results.get('title_meta', {})

    def _append_tm(label: str, key: str, *, include_length: bool = False) -> None:
        item = tm.get(key, {}) if isinstance(tm, dict) else {}
        if not isinstance(item, dict) or not item:
            return
        status = _status_text(item.get('status'))
        if not status:
            return
        sentence = f"{label} {status}"
        content = item.get('content')
        message = _clean_copy_value(item.get('message'))
        if include_length and content:
            sentence += f" at {len(content)} chars"
            content_text = _clean_copy_value(content)
            if content_text:
                sentence += f" (\"{content_text}\")"
        elif key == 'author' and content:
            sentence += f" ({_clean_copy_value(content)})"
        elif message:
            sentence += f" ({message})"
        fragments.append(sentence + ".")

    _append_tm("Title", "title", include_length=True)
    _append_tm("Meta description", "meta_description", include_length=True)
    _append_tm("Author", "author")

    hero = results.get('hero_image', {})
    if isinstance(hero, dict) and hero:
        status = _status_text(hero.get('status'))
        message = _clean_copy_value(hero.get('message'))
        src = _clean_copy_value(hero.get('src'))
        dims: List[str] = []
        if hero.get('width'):
            dims.append(str(hero.get('width')))
        if hero.get('height'):
            dims.append(str(hero.get('height')))
        sentence = f"Hero image {status}" if status else "Hero image status unknown"
        if dims:
            sentence += f" ({'x'.join(dims)}px)"
        if src:
            sentence += f" from {src}"
        if message:
            sentence += f" – {message}"
        fragments.append(sentence + ".")

    il = results.get('internal_links', {})
    if isinstance(il, dict) and il:
        status = _status_text(il.get('status'))
        broken = il.get('broken') or []
        message = _clean_copy_value(il.get('message'))
        checked = il.get('checked', 0)
        total = il.get('total_internal', 0)
        sentence = f"Internal links {status}" if status else "Internal links status unknown"
        sentence += f" (checked {checked} of {total})"
        if broken:
            sentence += f" with {len(broken)} broken links"
        if message:
            sentence += f" – {message}"
        fragments.append(sentence + ".")

    idx = results.get('indexability', {})
    if isinstance(idx, dict) and idx:
        status = _status_text(idx.get('status'))
        message = _clean_copy_value(idx.get('message'))
        if status or message:
            fragment = f"Indexability {status}" if status else "Indexability status unknown"
            if message:
                fragment += f" – {message}"
            fragments.append(fragment + ".")

    recs = _collect_recommendations(results)
    if recs:
        top_recs = recs[:3]
        rec_text = "; ".join(
            f"[{rec['status']}] {rec['area']}: {_clean_copy_value(rec['message'])}"
            for rec in top_recs
        )
        if len(recs) > 3:
            rec_text += "; additional action items logged"
        fragments.append(f"Key follow-ups: {rec_text}.")

    return " ".join(fragment.strip() for fragment in fragments if fragment).strip()


def format_results_as_text(results: Dict[str, Any], url: str) -> str:
    lines: List[str] = []
    add = lines.append

    add(f"URL: {url}")
    fetched = results.get('_fetched_at')
    if fetched:
        add(f"Fetched: {fetched}")

    section_scores = results.get('_section_scores') or {}
    if section_scores:
        add("")
        add("Section Scores")
        ordered_keys = [
            'title_meta', 'headings', 'schema', 'faq', 'mobile', 'robots', 'sitemaps',
            'internal_links', 'images', 'indexability', 'canonical', 'hreflang', 'locations', 'hero_image'
        ]
        for key in ordered_keys:
            if key in section_scores:
                label = key.replace('_', ' ').title()
                add(f"- {label}: {section_scores[key]:.1f}%")

    tm = results.get('title_meta', {})
    if tm:
        add("")
        add("Title & Meta")
        title = tm.get('title', {}) or {}
        title_detail = title.get('content') or title.get('message') or ''
        if title.get('content'):
            title_detail = f"\"{_clean_copy_value(title['content'])}\" ({len(title['content']):d} chars)"
        add(f"- Title: {_status_text(title.get('status'))} | { _clean_copy_value(title_detail) or 'No title found.'}")

        meta = tm.get('meta_description', {}) or {}
        meta_detail = meta.get('content') or meta.get('message') or ''
        if meta.get('content'):
            meta_detail = f"\"{_clean_copy_value(meta['content'])}\" ({len(meta['content']):d} chars)"
        add(f"- Meta Description: {_status_text(meta.get('status'))} | { _clean_copy_value(meta_detail) or 'No meta description found.'}")

        author = tm.get('author', {}) or {}
        author_detail = author.get('content') or author.get('message') or ''
        add(f"- Author: {_status_text(author.get('status'))} | { _clean_copy_value(author_detail) or 'No author meta tag detected.'}")

    hd = results.get('headings', {})
    if hd:
        add("")
        add("Headings")
        add(f"- H1: {_status_text(hd.get('h1_status'))} | {_clean_copy_value(hd.get('h1_content')) or 'No H1 found.'}")
        hierarchy_details = hd.get('h_hierarchy_message') or (
            f"Levels: {', '.join(str(lvl) for lvl in hd.get('h_tags_found', []))}" if hd.get('h_tags_found') else ''
        )
        add(f"- Hierarchy: {_status_text(hd.get('h_hierarchy'))} | {_clean_copy_value(hierarchy_details) or 'Hierarchy issues detected.'}")

    sc = results.get('schema', {})
    if sc:
        add("")
        add("Schema")
        schema_status = _status_text('pass' if sc.get('schema_found') else 'fail')
        types = ', '.join(sorted({str(t) for t in sc.get('types', [])}))
        add(f"- Structured Data: {schema_status} | Types: {_clean_copy_value(types) or 'None'}")
        url_rows = sc.get('urls') or []
        if url_rows:
            add(f"- Script URLs: {', '.join(url_rows[:5])}{'…' if len(url_rows) > 5 else ''}")

    faq = results.get('faq', {})
    if faq:
        add("")
        add("FAQ")
        add(f"- Status: {_status_text(faq.get('status'))} | {_clean_copy_value(faq.get('message')) or 'No FAQ notes.'}")
        add(f"- H3 Count: {faq.get('h3_count', 0)}")
        add(f"- H5 Count: {faq.get('h5_count', 0)}")

    mobile = results.get('mobile_responsiveness', {})
    if mobile:
        add("")
        add("Mobile Responsiveness")
        add(f"- Status: {_status_text(mobile.get('status'))} | {_clean_copy_value(mobile.get('message')) or 'No mobile warnings.'}")
        breakpoints = (mobile.get('breakpoints') or {}).get('widths', {})
        if breakpoints:
            bp_parts = []
            for label in ('mobile', 'tablet', 'desktop'):
                widths = breakpoints.get(label, [])
                if widths:
                    bp_parts.append(f"{label}: {', '.join(str(w) for w in widths[:4])}{'…' if len(widths) > 4 else ''}")
            if bp_parts:
                add(f"- Breakpoints: {' | '.join(bp_parts)}")

    idx = results.get('indexability', {})
    if idx:
        add("")
        add("Indexability")
        add(f"- Status: {_status_text(idx.get('status'))} | {_clean_copy_value(idx.get('message')) or 'No indexability warnings.'}")
        add(f"- Meta Robots: {_clean_copy_value(idx.get('meta_robots')) or '-'}")
        add(f"- X-Robots-Tag: {_clean_copy_value(idx.get('x_robots_tag')) or '-'}")

    robots = (results.get('robots_sitemaps') or {}).get('robots', {})
    sitemaps = (results.get('robots_sitemaps') or {}).get('sitemaps', {})
    add("")
    add("Robots & Sitemaps")
    add(f"- Robots.txt: {'Present' if robots.get('present') else 'Missing'} | {_clean_copy_value(robots.get('url')) or 'No URL recorded.'}")
    add(f"- Sitemaps Status: {_status_text(sitemaps.get('status'))} | {_clean_copy_value(sitemaps.get('message')) or 'No sitemap message.'}")
    validated = sitemaps.get('validated') or []
    if validated:
        urls = [v.get('sitemap_url') for v in validated[:5] if v.get('sitemap_url')]
        if urls:
            add(f"- Sitemap URLs: {', '.join(urls)}{', …' if len(validated) > 5 else ''}")

    il = results.get('internal_links', {})
    if il:
        add("")
        add("Internal Links")
        add(
            f"- Status: {_status_text(il.get('status'))} | Checked {il.get('checked', 0)} of {il.get('total_internal', 0)} links; context links: {il.get('contextual_links', 0)}"
        )
        add(f"- Notes: {_clean_copy_value(il.get('message')) or 'No internal link issues.'}")
        broken = il.get('broken') or []
        if broken:
            preview = ', '.join(broken[:5])
            if len(broken) > 5:
                preview += ', …'
            add(f"- Broken Links: {preview}")

    images = results.get('images', {})
    if images:
        add("")
        add("Images")
        total_visible = images.get('total_images', 0)
        skipped_hidden = images.get('skipped_hidden', 0)
        add(f"- Status: {_status_text(images.get('status'))} | {_clean_copy_value(images.get('message')) or 'No image issues.'}")
        add(f"- Visible Images Audited: {total_visible}")
        if skipped_hidden:
            add(f"- Hidden/Lazy Images Skipped: {skipped_hidden}")
        miss_with_title = images.get('missing_alt_with_title') or []
        if miss_with_title:
            preview = ', '.join(miss_with_title[:5]) + ('…' if len(miss_with_title) > 5 else '')
            add(f"- Missing alt (image has title attribute): {preview}")

    ch = results.get('canonical_hreflang', {})
    canonical = ch.get('canonical', {}) if isinstance(ch, dict) else {}
    hreflang = ch.get('hreflang', {}) if isinstance(ch, dict) else {}
    add("")
    add("Canonical & Hreflang")
    add(f"- Canonical: {_status_text(canonical.get('status'))} | {_clean_copy_value(canonical.get('url')) or 'No canonical tag found.'}")
    add(f"- Notes: {_clean_copy_value(canonical.get('message')) or 'No canonical warnings.'}")
    hre_entries = hreflang.get('entries') or []
    if hre_entries:
        preview = ', '.join(f"{entry.get('lang')}: {entry.get('url')}" for entry in hre_entries[:5])
        if len(hre_entries) > 5:
            preview += ', …'
        add(f"- Hreflang Entries: {preview}")
    else:
        add(f"- Hreflang: {_status_text(hreflang.get('status'))} | {_clean_copy_value(hreflang.get('message')) or 'No hreflang tags present.'}")

    locations = results.get('locations', {})
    if locations:
        add("")
        add("Locations")
        add(f"- Status: {_status_text(locations.get('status'))} | {_clean_copy_value(locations.get('message')) or 'No location issues.'}")
        addresses = locations.get('addresses') or []
        if addresses:
            preview = '; '.join(addresses[:5])
            if len(addresses) > 5:
                preview += '; …'
            add(f"- Addresses: {preview}")

    hero = results.get('hero_image', {})
    if hero:
        add("")
        add("Hero Image")
        add(f"- Status: {_status_text(hero.get('status'))} | {_clean_copy_value(hero.get('message')) or 'Hero image is configured correctly.'}")
        hero_src = hero.get('src')
        if hero_src:
            add(f"- Source: {hero_src}")
        hero_size = []
        if hero.get('width'):
            hero_size.append(str(hero.get('width')))
        if hero.get('height'):
            hero_size.append(str(hero.get('height')))
        if hero_size:
            add(f"- Dimensions: {'x'.join(hero_size)}")

    recommendations = _collect_recommendations(results)
    if recommendations:
        add("")
        add("Action Items")
        for rec in recommendations:
            add(f"- [{rec['status'].upper()}] {rec['area']}: {_clean_copy_value(rec['message'])}")

    return "\n".join(lines).rstrip() + "\n"


def print_results_as_text(results: Dict[str, Any], url: str) -> None:
    report = format_results_as_text(results, url)
    print("\n=== Copy-Friendly Audit Report ===")
    print(report)

def print_results_as_tables(results: Dict[str, Any], url: str):
    # Per-section scores (percent)
    section_scores = results.get('_section_scores')
    if section_scores:
        rows = [[k.replace('_', ' ').title(), f"{v:.1f}%"] for k, v in section_scores.items()]
        _print_table("Section Scores", ["Item", "Percent"], rows)

    # Title & Meta
    tm = results.get('title_meta', {})
    _print_table(
        "Title & Meta",
        ["Item", "Status", "Chars", "Percent", "Content/Message"],
        [
            [
                "Title",
                tm.get('title', {}).get('status'),
                len(tm.get('title', {}).get('content') or "") if tm.get('title') else "",
                f"{_status_to_percent(tm.get('title', {}).get('status')):.0f}%",
                _truncate(tm.get('title', {}).get('content') or tm.get('title', {}).get('message', "")),
            ],
            [
                "Meta Description",
                tm.get('meta_description', {}).get('status'),
                len(tm.get('meta_description', {}).get('content') or "") if tm.get('meta_description') else "",
                f"{_status_to_percent(tm.get('meta_description', {}).get('status')):.0f}%",
                _truncate(tm.get('meta_description', {}).get('content') or tm.get('meta_description', {}).get('message', "")),
            ],
            [
                "Author",
                tm.get('author', {}).get('status'),
                len(tm.get('author', {}).get('content') or "") if tm.get('author') else "",
                f"{_status_to_percent(tm.get('author', {}).get('status')):.0f}%",
                _truncate(tm.get('author', {}).get('content') or tm.get('author', {}).get('message', "")),
            ],
        ],
    )

    # Headings
    hd = results.get('headings', {})
    _print_table(
        "Headings",
        ["Item", "Status", "Percent", "Details"],
        [
            [
                "H1",
                hd.get('h1_status'),
                f"{_status_to_percent(hd.get('h1_status')):.0f}%",
                _truncate(hd.get('h1_content', "")) or "",
            ],
            [
                "Hierarchy",
                hd.get('h_hierarchy'),
                f"{_status_to_percent(hd.get('h_hierarchy')):.0f}%",
                "Levels: " + ",".join(map(str, hd.get('h_tags_found', []))) if hd.get('h_tags_found') else "",
            ],
        ],
    )

    # Schema
    sc = results.get('schema', {})
    _print_table(
        "Schema (JSON-LD)",
        ["Status", "Percent", "Blocks", "Types"],
        [["pass" if sc.get('schema_found') else "fail", f"{(100.0 if sc.get('schema_found') else 0.0):.0f}%", len(sc.get('schemas', [])), ", ".join(sorted(set([str(t) for t in sc.get('types', [])]))[:6])]],
    )
    if sc.get('urls'):
        _print_table(
            "Schema Script SRC",
            ["Source URL"],
            [[_truncate(u, 120)] for u in sc.get('urls', [])[:10]],
        )
    # FAQ
    faq = results.get('faq', {})
    _print_table(
        "FAQ",
        ["H3 Present", "H5 Present", "Schema FAQPage", "Status", "Message"],
        [[
            "yes" if faq.get('h3_count', 0) > 0 else "no",
            "yes" if faq.get('h5_count', 0) > 0 else "no",
            "yes" if sc.get('faqpage_found') else "no",
            faq.get('status', ''),
            _truncate(faq.get('message', '')),
        ]],
    )
    if faq.get('h5_examples'):
        _print_table(
            "FAQ H5 Examples",
            ["Heading Text"],
            [[_truncate(text, 120)] for text in faq.get('h5_examples', [])[:10]],
        )

    # Mobile
    mb = results.get('mobile_responsiveness', {})
    _print_table(
        "Mobile Responsiveness",
        ["Status", "Percent", "Message"],
        [[mb.get('status'), f"{_status_to_percent(mb.get('status')):.0f}%", _truncate(mb.get('message', ""))]],
    )
    # Responsive breakpoints (if available)
    bp = (mb or {}).get('breakpoints') or {}
    if bp:
        _print_table(
            "Responsive Breakpoints",
            ["Category", "Found", "Count", "Widths(px)"],
            [
                [
                    "Mobile",
                    "yes" if (bp.get('found', {}).get('mobile')) else "no",
                    bp.get('counts', {}).get('mobile', 0),
                    ", ".join(map(str, bp.get('widths', {}).get('mobile', [])))
                ],
                [
                    "Tablet",
                    "yes" if (bp.get('found', {}).get('tablet')) else "no",
                    bp.get('counts', {}).get('tablet', 0),
                    ", ".join(map(str, bp.get('widths', {}).get('tablet', [])))
                ],
                [
                    "Desktop",
                    "yes" if (bp.get('found', {}).get('desktop')) else "no",
                    bp.get('counts', {}).get('desktop', 0),
                    ", ".join(map(str, bp.get('widths', {}).get('desktop', [])))
                ],
            ],
        )

    # Indexability
    ix = results.get('indexability', {})
    _print_table(
        "Indexability",
        ["Status", "Percent", "Meta", "Header", "Message"],
        [[ix.get('status', ''), f"{_status_to_percent(ix.get('status')):.0f}%", _truncate(ix.get('meta_robots', '') or ''), _truncate(ix.get('x_robots_tag', '') or ''), _truncate(ix.get('message', ''))]],
    )

    # Locations
    loc = results.get('locations', {})
    _print_table(
        "Locations",
        ["Status", "Percent", "Count", "Message"],
        [[loc.get('status', ''), f"{_status_to_percent(loc.get('status')):.0f}%", loc.get('count', 0), _truncate(loc.get('message', ''))]],
    )
    if loc.get('addresses'):
        _print_table(
            "Addresses",
            ["Address"],
            [[_truncate(a, 120)] for a in (loc.get('addresses') or [])[:20]],
        )

    hero = results.get('hero_image', {})
    if hero:
        _print_table(
            "Hero Image",
            ["Status", "Width", "Height", "Src", "Message"],
            [[
                hero.get('status', ''),
                hero.get('width', ''),
                hero.get('height', ''),
                _truncate(hero.get('src', ''), 80),
                _truncate(hero.get('message', '')),
            ]],
        )

    # Robots & Sitemaps
    rs = results.get('robots_sitemaps', {})
    robots_present = rs.get('robots', {}).get('present')
    _print_table(
        "Robots",
        ["Present", "Percent", "URL"],
        [["yes" if robots_present else "no", f"{(100.0 if robots_present else 0.0):.0f}%", rs.get('robots', {}).get('url', "")]],
    )
    val = rs.get('sitemaps', {}).get('validated', [])
    _print_table(
        "Sitemaps",
        ["Status", "Percent", "URL", "Message"],
        (
            [[
                v.get('status'),
                f"{_status_to_percent(v.get('status')):.0f}%",
                v.get('sitemap_url'),
                v.get('message')
            ] for v in (val if val else [])]
            or [[rs.get('sitemaps', {}).get('status', 'fail'), f"{_status_to_percent(rs.get('sitemaps', {}).get('status')):.0f}%", '', 'No sitemaps validated']]
        ),
    )

    # Internal Links
    il = results.get('internal_links', {})
    checked = int(il.get('checked') or 0)
    broken_ct = len(il.get('broken') or [])
    il_percent = (100.0 * (1.0 - broken_ct / float(checked))) if checked > 0 else 0.0
    _print_table(
        "Internal Links",
        ["Total", "Checked", "Contextual", "Status", "Percent", "Message"],
        [[il.get('total_internal', 0), checked, il.get('contextual_links', 0), il.get('status', ''), f"{il_percent:.0f}%", _truncate(il.get('message', ''))]],
    )
    if il.get('broken'):
        _print_table(
            "Broken Internal Links",
            ["Link"],
            [[_truncate(link, 120)] for link in il.get('broken', [])[:20]],
        )

    # Images
    im = results.get('images', {})
    total_imgs = int(im.get('total_images') or 0)
    miss_ct = len(im.get('missing_alt') or [])
    if total_imgs == 0:
        total_found = int(im.get('total_found') or 0)
        hidden = int(im.get('skipped_hidden') or 0)
        img_percent = 100.0 if total_found > 0 and hidden == total_found else 0.0
    else:
        img_percent = 100.0 * (1.0 - miss_ct / float(total_imgs))
    _print_table(
        "Images",
        ["Total", "Status", "Percent", "Message"],
        [[total_imgs, im.get('status', ''), f"{img_percent:.0f}%", _truncate(im.get('message', ''))]],
    )
    # Combine all alt-related issues into a single table
    issues_rows: List[List[str]] = []
    for src in (im.get('missing_alt_with_title') or [])[:20]:
        issues_rows.append(["Missing alt (has title)", _truncate(src, 120)])
    if issues_rows:
        _print_table(
            "Image Alt Issues",
            ["Issue", "Src/Alt"],
            issues_rows,
        )
    # Image sizes (if available)
    sizes = im.get('sizes') or []
    if sizes:
        _print_table(
            "Image Sizes",
            ["Src", "Size (KB)", "Type", "Display"],
            [[_truncate(it.get('src', ''), 80), f"{(it.get('bytes') or 0)/1024:.0f}", (it.get('type') or ''), (it.get('display') or '')] for it in sizes[:20]],
        )
        if (im.get('oversized_count') or 0) > 0:
            thr_kb = int((im.get('oversized_threshold_bytes') or 0)/1024)
            _print_table(
                "Image Size Notes",
                ["Oversized (>KB)", "Count"],
                [[thr_kb, im.get('oversized_count')]],
            )

    # Canonical & Hreflang
    ch = results.get('canonical_hreflang', {})
    can = ch.get('canonical', {})
    _print_table(
        "Canonical",
        ["Status", "Percent", "Message", "URL", "Multiple"],
        [[can.get('status'), f"{_status_to_percent(can.get('status')):.0f}%", _truncate(can.get('message', '')), _truncate(can.get('url', '')), str(can.get('multiple', False))]],
    )
    hre = ch.get('hreflang', {})
    entries = hre.get('entries', [])
    _print_table(
        "Hreflang Entries",
        ["Lang", "URL"],
        [[e.get('lang'), _truncate(e.get('url', ''), 120)] for e in entries[:20]] or [["-", "-"]],
    )
    if hre.get('duplicates') or hre.get('invalid'):
        _print_table(
            "Hreflang Notes",
            ["Status", "Percent", "Duplicates", "Invalid"],
            [[hre.get('status'), f"{_status_to_percent(hre.get('status')):.0f}%", ",".join(hre.get('duplicates', [])), ",".join(hre.get('invalid', []))]],
        )

    _print_recommendations(results)

    summary = _build_summary_paragraph(results, url)
    if summary:
        if _rich_enabled():
            _RICH_CONSOLE.print(Panel.fit(summary, title="Summary", border_style="cyan"))
        else:
            print("\nSummary:\n" + summary + "\n")

    # Spelling and AI sections removed

def _append_history(history_path: str, entries: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def _show_history(history_path: str, limit: int = 20):
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"No history file found at {history_path}")
        return
    # Take last N
    items = []
    for line in lines[-limit:]:
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = []
    for it in items:
        summ = it.get("score_summary", {})
        rows.append([
            it.get("timestamp", ""),
            _truncate(it.get("url", ""), 80),
            f"{summ.get('percent', 0):.1f}%",
            summ.get("result", ""),
        ])
    _print_table("History (most recent)", ["Timestamp", "URL", "%", "Result"], rows)

    # Per-section table (if section scores available)
    sec_rows = []
    for it in items:
        secs = it.get("section_scores", {})
        if not secs:
            continue
        sec_rows.append([
            it.get("timestamp", ""),
            _truncate(it.get("url", ""), 60),
            f"{secs.get('title_meta', 0):.0f}%",
            f"{secs.get('headings', 0):.0f}%",
            f"{secs.get('schema', 0):.0f}%",
            f"{secs.get('mobile', 0):.0f}%",
            f"{secs.get('robots', 0):.0f}%",
            f"{secs.get('sitemaps', 0):.0f}%",
            f"{secs.get('internal_links', 0):.0f}%",
            f"{secs.get('images', 0):.0f}%",
            f"{secs.get('indexability', 0):.0f}%",
            f"{secs.get('canonical', 0):.0f}%",
            f"{secs.get('hreflang', 0):.0f}%",
            f"{secs.get('locations', 0):.0f}%",
        ])
    if sec_rows:
        _print_table(
            "History Section Scores",
            [
                "Timestamp",
                "URL",
                "Title/Meta",
                "Headings",
                "Schema",
                "Mobile",
                "Robots",
                "Sitemaps",
                "Internal",
                "Images",
                "Index",
                "Canonical",
                "Hreflang",
                "Locations",
            ],
            sec_rows,
        )




def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if getattr(args, 'proxy', None):
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
    if getattr(args, 'ca_bundle', None):
        os.environ['REQUESTS_CA_BUNDLE'] = args.ca_bundle
    if getattr(args, 'insecure', False):
        os.environ['SEO_CHECKER_INSECURE'] = '1'

    if args.show_history is not None:
        _show_history(args.history_file, args.show_history)
        return 0

    urls: List[str] = list(args.urls)
    if args.url_file:
        try:
            with open(args.url_file, 'r', encoding='utf-8') as handle:
                urls.extend([
                    line.strip()
                    for line in handle
                    if line.strip() and not line.strip().startswith('#')
                ])
        except Exception as exc:
            print(f"Warning: could not read --url-file: {exc}")
    if args.child_url:
        urls.append(args.child_url)

    seen: Set[str] = set()
    unique_urls: List[str] = []
    for candidate in urls:
        if candidate not in seen:
            unique_urls.append(candidate)
            seen.add(candidate)

    if not unique_urls:
        print(
            'No URLs provided. Supply one or more URLs, use --url-file, or run with --show-history.'
        )
        return 2

    outputs: List[Dict[str, Any]] = []
    show_table = args.format in ('table', 'both', 'table+text')
    show_text = args.format in ('text', 'table+text', 'text+json')
    show_json = args.format in ('json', 'both', 'text+json')
    for site_url in unique_urls:
        is_child = bool(args.child_url and site_url == args.child_url)
        result = run_all_checks(
            site_url,
            timeout=args.timeout,
            use_scraperapi=args.use_scraperapi,
            max_links=args.max_links,
            quiet=(args.quiet or is_child),
            keyword=args.keyword,
        )
        result['_score_summary'] = compute_score(result, threshold=args.threshold)
        result['_section_scores'] = compute_section_scores(result)
        outputs.append({'url': site_url, 'results': result})

        if not args.quiet and show_table and not is_child:
            print_results_as_tables(result, site_url)
        if not args.quiet and show_text and not is_child:
            print_results_as_text(result, site_url)

        if args.capture_screenshots and not is_child:
            ok, message = _run_puppeteer_capture(
                args.puppeteer_script,
                site_url,
                args.screenshot_dir,
                args.screenshot_clip,
            )
            if not args.quiet:
                if ok:
                    print(colorize(message, 'info'))
                else:
                    print(colorize(f"Puppeteer capture failed: {message}", 'warning'))

    if not args.quiet and len(outputs) == 2 and show_table:
        parent, child = outputs[0], outputs[1]
        _print_format_comparison(
            parent['url'], parent['results'], child['url'], child['results']
        )

    if not args.quiet and show_json:
        print("\n--- SEO Check Results (JSON) ---")
        print(json.dumps(outputs, indent=2))

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(outputs, handle, ensure_ascii=False, indent=2)

    if args.output_csv:
        import csv

        with open(args.output_csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['url', 'check', 'status', 'message'])
            for item in outputs:
                url = item['url']
                result = item['results']

                tm = result.get('title_meta', {})
                writer.writerow([
                    url,
                    'title',
                    tm.get('title', {}).get('status'),
                    tm.get('title', {}).get('message', ''),
                ])
                writer.writerow([
                    url,
                    'meta_description',
                    tm.get('meta_description', {}).get('status'),
                    tm.get('meta_description', {}).get('message', ''),
                ])
                writer.writerow([
                    url,
                    'author',
                    tm.get('author', {}).get('status'),
                    tm.get('author', {}).get('content', '')
                    or tm.get('author', {}).get('message', ''),
                ])

                hd = result.get('headings', {})
                writer.writerow([url, 'h1', hd.get('h1_status'), hd.get('h_hierarchy')])

                schema_info = result.get('schema', {})
                writer.writerow([
                    url,
                    'schema',
                    'pass' if schema_info.get('schema_found') else 'fail',
                    f"{len(schema_info.get('schemas', []))} blocks",
                ])

                mobile_info = result.get('mobile_responsiveness', {})
                writer.writerow([
                    url,
                    'mobile',
                    mobile_info.get('status'),
                    mobile_info.get('message', ''),
                ])

                robots = result.get('robots_sitemaps', {})
                writer.writerow([
                    url,
                    'robots',
                    'pass' if robots.get('robots', {}).get('present') else 'fail',
                    robots.get('robots', {}).get('url') or '',
                ])
                writer.writerow([
                    url,
                    'sitemaps',
                    robots.get('sitemaps', {}).get('status'),
                    ','.join(
                        [
                            entry.get('sitemap_url', '')
                            for entry in robots.get('sitemaps', {}).get('validated', [])
                        ]
                    ),
                ])

                internal_links_info = result.get('internal_links', {})
                writer.writerow([
                    url,
                    'internal_links',
                    internal_links_info.get('status'),
                    internal_links_info.get('message'),
                ])

                image_info = result.get('images', {})
                writer.writerow([
                    url,
                    'images',
                    image_info.get('status'),
                    image_info.get('message'),
                ])

                indexability_info = result.get('indexability', {})
                writer.writerow([
                    url,
                    'indexability',
                    indexability_info.get('status'),
                    indexability_info.get('message'),
                ])

                faq_info = result.get('faq', {})
                writer.writerow([url, 'faq', faq_info.get('status'), faq_info.get('message')])

                locations_info = result.get('locations', {})
                writer.writerow([
                    url,
                    'locations',
                    locations_info.get('status'),
                    f"{locations_info.get('count', 0)} addresses",
                ])
                writer.writerow([
                    url,
                    'locations',
                    locations_info.get('status'),
                    locations_info.get('message'),
                ])

                canonical_info = result.get('canonical_hreflang', {})
                writer.writerow([
                    url,
                    'canonical',
                    canonical_info.get('canonical', {}).get('status'),
                    canonical_info.get('canonical', {}).get('message'),
                ])
                writer.writerow([
                    url,
                    'hreflang',
                    canonical_info.get('hreflang', {}).get('status'),
                    canonical_info.get('hreflang', {}).get('message'),
                ])

    if not args.no_history:
        history_entries: List[Dict[str, Any]] = []
        timestamp = (
            datetime.now(timezone.utc)
            .isoformat(timespec='seconds')
            .replace('+00:00', 'Z')
        )
        for item in outputs:
            result = item['results']
            history_entries.append(
                {
                    'timestamp': timestamp,
                    'url': item['url'],
                    'score_summary': result.get('_score_summary', {}),
                    'section_scores': result.get('_section_scores', {}),
                }
            )
        _append_history(args.history_file, history_entries)

    any_fetch_error = any(item['results'].get('error') for item in outputs)
    any_below_threshold = any(
        item['results'].get('_score_summary', {}).get('percent', 0.0) < args.threshold
        for item in outputs
    )
    any_nonpass = any(_has_nonpass_status(item['results']) for item in outputs)

    if any_fetch_error:
        return 2
    if any_below_threshold or any_nonpass:
        return 1
    return 0


def console_entry() -> None:  # pragma: no cover - CLI entry point
    sys.exit(main())


if __name__ == '__main__':  # pragma: no cover - CLI entry point
    console_entry()

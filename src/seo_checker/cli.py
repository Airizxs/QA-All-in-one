import os
import json
import sys
import argparse
import re
import threading
import time
import itertools
import subprocess
import html
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
    content_imgs = int(im.get('content_image_count') or 0)
    issue_refs: Set[str] = set()
    for ref in (im.get('content_missing_alt') or []):
        if isinstance(ref, str) and ref:
            issue_refs.add(ref)
    for ref in (im.get('content_missing_title') or []):
        if isinstance(ref, str) and ref:
            issue_refs.add(ref)
    issue_count = len(issue_refs)
    if content_imgs == 0:
        total_visible = int(im.get('total_images') or 0)
        total_found = int(im.get('total_found') or 0)
        hidden = int(im.get('skipped_hidden') or 0)
        if total_visible == 0 and total_found > 0 and hidden == total_found:
            section_scores['images'] = 100.0
        else:
            section_scores['images'] = 0.0
    else:
        section_scores['images'] = max(0.0, 100.0 * (1.0 - issue_count / float(content_imgs)))

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


def _status_to_emoji(status: Optional[str]) -> str:
    if not status:
        return "⚠️"
    normalized = str(status).strip().lower()
    if normalized in _PASS_STATUSES:
        return "✅"
    if normalized in {"warn", "warning"}:
        return "⚠️"
    if normalized in {"info"}:
        return "ℹ️"
    return "❌"


def _status_severity(status: Optional[str]) -> int:
    if not status:
        return 1
    normalized = str(status).strip().lower()
    if normalized in {"fail", "missing", "error", "no"}:
        return 0
    if normalized in {"warn", "warning"}:
        return 1
    return 2


def _collect_qa_summary_rows(results: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    if not isinstance(results, dict):
        return []

    error_message = _clean_copy_value(results.get("error"))
    if error_message:
        safe_error = _clean_copy_value(error_message)
        return [("Audit", "Fail", safe_error or "Request failed.")]

    recommendations = _collect_recommendations(results)
    rec_by_area: Dict[str, List[Dict[str, str]]] = {}
    for rec in recommendations:
        rec_by_area.setdefault(rec["area"], []).append(rec)

    notes_map: Dict[str, List[str]] = {}

    def append_notes(area: str, raw_notes: List[str]) -> None:
        if not raw_notes:
            return
        existing = notes_map.get(area, [])
        for note in raw_notes:
            cleaned = _clean_copy_value(note)
            if cleaned and cleaned not in existing:
                existing.append(cleaned)
        if existing:
            notes_map[area] = existing

    def _collect_item_notes(item: Dict[str, Any]) -> List[str]:
        if not isinstance(item, dict):
            return []
        notes: List[str] = []
        raw_notes = item.get("notes")
        if isinstance(raw_notes, (list, tuple, set)):
            notes.extend(str(n) for n in raw_notes if n)
        elif raw_notes:
            notes.append(str(raw_notes))
        message = item.get("message")
        status = (item.get("status") or "").strip().lower()
        if message and (not status or status in _PASS_STATUSES):
            notes.append(message)
        return notes

    def gather(area_keys: List[str]) -> Optional[Dict[str, Any]]:
        matches: List[Dict[str, str]] = []
        for key in area_keys:
            matches.extend(rec_by_area.get(key, []))
        if not matches:
            return None
        matches.sort(key=lambda rec: _status_severity(rec.get("status")))
        status = matches[0].get("status")
        # Preserve insertion order while removing duplicates
        deduped_messages: Dict[str, None] = {}
        for rec in matches:
            message = _clean_copy_value(rec.get("message"))
            if message:
                deduped_messages[message] = None
        message_text = "; ".join(deduped_messages.keys()) or _clean_copy_value(matches[0].get("message"))
        return {
            "status": status,
            "message": message_text,
        }

    tm = results.get("title_meta", {}) or {}
    append_notes("Title & Meta", _collect_item_notes(tm.get("title", {}) or {}))
    append_notes("Title & Meta", _collect_item_notes(tm.get("meta_description", {}) or {}))

    faq = results.get("faq", {}) or {}
    append_notes("FAQ", _collect_item_notes(faq))

    indexability = results.get("indexability", {}) or {}
    append_notes("Indexability", _collect_item_notes(indexability))

    canonical_bundle = results.get("canonical_hreflang", {}) or {}
    canonical = canonical_bundle.get("canonical", {}) or {}
    append_notes("Canonical", _collect_item_notes(canonical))

    def success_title_meta() -> str:
        return "Optimized and aligned with keywords"

    def success_headings() -> str:
        hd = results.get("headings", {}) or {}
        parts: List[str] = []
        h1_text = _clean_copy_value(hd.get("h1_content"))
        if h1_text:
            parts.append(f'H1 "{_truncate(h1_text, 72)}"')
        hierarchy = hd.get("h_tags_found") or []
        if hierarchy:
            levels = ", ".join(str(level) for level in hierarchy[:6])
            parts.append(f"levels {levels}")
        detail = "; ".join(parts)
        if detail:
            return "Heading structure is in place"
        return "Heading structure is in place"

    def success_internal_links() -> str:
        il = results.get("internal_links", {}) or {}
        checked = int(il.get("checked") or 0)
        contextual = int(il.get("contextual_links") or 0)
        message_parts = [f"checked {checked} links with no broken URLs detected"]
        if contextual:
            message_parts.append(f"{contextual} contextual links found")
        return "Internal linking validated"

    def success_schema() -> str:
        schema = results.get("schema", {}) or {}
        types = schema.get("types") or []
        unique_types = sorted({str(t) for t in types if t})
        type_text = ", ".join(unique_types[:5])
        block_count = len(schema.get("schemas") or [])
        parts: List[str] = []
        if block_count:
            parts.append(f"{block_count} structured data block{'s' if block_count != 1 else ''}")
        if type_text:
            parts.append(f"types: {type_text}{'…' if len(unique_types) > 5 else ''}")
        return "Schema markup present"

    def success_images() -> str:
        images = results.get("images", {}) or {}
        pairs = images.get("content_images_with_alt_title") or []
        previews: List[str] = []
        for item in pairs:
            if not isinstance(item, dict):
                continue
            alt_text = _clean_copy_value(item.get("alt"))
            title_text = _clean_copy_value(item.get("title"))
            snippet_parts: List[str] = []
            if alt_text:
                snippet_parts.append(f"alt: {alt_text}")
            if title_text:
                snippet_parts.append(f"title: {title_text}")
            snippet = " | ".join(snippet_parts)
            if snippet:
                previews.append(snippet)
            if len(previews) >= 3:
                break
        if previews:
            preview_text = "; ".join(previews)
            remaining = max(0, len(pairs) - len(previews))
            if remaining > 0:
                preview_text += "…"
            return f"Photo alt/title pairs ok ({preview_text})"
        message = _clean_copy_value(images.get("message"))
        if message:
            return message
        total_visible = int(images.get("total_images") or 0)
        return f"Audited {total_visible} images"

    def success_mobile() -> str:
        mobile = results.get("mobile_responsiveness", {}) or {}
        message = _clean_copy_value(mobile.get("message"))
        if message:
            return message
        breakpoints = (mobile.get("breakpoints") or {}).get("widths", {})
        bp_parts = []
        for label in ("mobile", "tablet", "desktop"):
            widths = breakpoints.get(label, [])
            if widths:
                bp_parts.append(f"{label}: {', '.join(str(w) for w in widths[:3])}{'…' if len(widths) > 3 else ''}")
        if bp_parts:
            return "Responsive across key breakpoints"
        return "Responsive across devices"

    def success_faq() -> str:
        if faq:
            message = _clean_copy_value(faq.get("message"))
            if message:
                return message
            if faq.get("status"):
                return "FAQ content meets requirements"
        return "FAQ section reviewed"

    def success_indexability() -> str:
        if indexability:
            message = _clean_copy_value(indexability.get("message"))
            if message:
                return message
        return "Index directives allow crawling"

    def success_canonical() -> str:
        canonical_url = _clean_copy_value(canonical.get("url"))
        if canonical_url:
            return f"Canonical points to {canonical_url}"
        message = _clean_copy_value(canonical.get("message"))
        if message:
            return message
        return "Canonical configuration reviewed"

    section_map = [
        ("Title & Meta", ["Title Tag", "Meta Description", "Author Meta"], success_title_meta),
        ("Headings", ["H1 Heading", "Heading Hierarchy"], success_headings),
        ("Internal Links", ["Internal Links"], success_internal_links),
        ("Schema", ["Schema Markup"], success_schema),
        ("Image Optimization", ["Images"], success_images),
        ("Responsiveness", ["Mobile"], success_mobile),
        ("FAQ", ["FAQ"], success_faq),
        ("Indexability", ["Indexability"], success_indexability),
        ("Canonical", ["Canonical"], success_canonical),
    ]

    rows: List[Tuple[str, str, str]] = []
    area_alias = {
        "Title & Meta": "Title & Meta",
        "Headings": "Headings",
        "Internal Links": "Internal Links",
        "Schema": "Schema",
        "Image Optimization": "Image Optimization",
        "Responsiveness": "Responsiveness",
        "FAQ": "FAQ",
        "Indexability": "Indexability",
        "Canonical": "Canonical",
    }
    def _normalize_sentences(parts: List[str]) -> List[str]:
        normalized: List[str] = []
        for part in parts:
            text = _clean_copy_value(part)
            if not text:
                continue
            if text.endswith(('.', '!', '?')):
                normalized.append(text)
            else:
                normalized.append(f"{text}.")
        seen: Set[str] = set()
        ordered: List[str] = []
        for text in normalized:
            if text not in seen:
                seen.add(text)
                ordered.append(text)
        return ordered

    for label, rec_keys, success_fn in section_map:
        rec_info = gather(rec_keys)
        raw_section_notes = notes_map.get(label, [])
        section_notes = _normalize_sentences(raw_section_notes)

        if rec_info:
            status_value = _clean_copy_value(rec_info.get("status")) or "fail"
            pass_fail = "Pass" if status_value.lower() in _PASS_STATUSES else "Fail"
            detail_parts = _normalize_sentences([rec_info.get("message", "")])
        else:
            pass_fail = "Pass"
            detail_parts = _normalize_sentences([success_fn()])

        detail_parts.extend(section_notes)
        combined_detail = " ".join(detail_parts).strip()

        display_area = area_alias.get(label, label)
        rows.append((display_area, pass_fail, combined_detail or "-"))

    return rows


def _shorten_summary_detail(detail: str) -> str:
    detail = (detail or "").strip()
    if not detail:
        return detail

    has_url = any(token in detail for token in ("http://", "https://", "www."))

    cut_idx: Optional[int] = None
    if not has_url:
        for sep in (". ", "; ", "! ", "? "):
            idx = detail.find(sep)
            if idx != -1:
                cut_idx = idx
                break
    else:
        # Keep the sentence containing the URL intact
        url_idx = min(
            (detail.find(token) for token in ("http://", "https://", "www.") if detail.find(token) != -1),
            default=-1,
        )
        if url_idx == -1:
            for sep in (". ", "; ", "! ", "? "):
                idx = detail.find(sep)
                if idx != -1:
                    cut_idx = idx
                    break
        else:
            for sep in (". ", "; ", "! ", "? "):
                idx = detail.find(sep, url_idx)
                if idx != -1:
                    cut_idx = idx
                    break

    if cut_idx is not None:
        detail = detail[:cut_idx]
    else:
        for end_sep in (".", ";", "!", "?"):
            if detail.endswith(end_sep):
                detail = detail[: -1]
                break

    max_len = 90
    if len(detail) > max_len:
        if has_url:
            return detail.strip()
        detail = detail[: max_len - 3].rstrip() + "..."
    return detail.strip()


def _build_qa_comment_lines(results: Dict[str, Any]) -> List[str]:
    rows = _collect_qa_summary_rows(results)
    if not rows:
        return []

    display_map = {
        "Title & Meta": "Title and Meta",
        "Headings": "Headings",
        "Internal Links": "Internal Links",
        "Schema": "Schema",
        "Image Optimization": "Image",
        "Responsiveness": "Responsive",
        "FAQ": "FAQ",
        "Indexability": "Indexability",
        "Canonical": "Canonical",
    }

    pass_rows: List[Tuple[str, str]] = []
    fail_rows: List[Tuple[str, str]] = []

    for area, status, detail in rows:
        label = display_map.get(area, area)
        detail_clean = _clean_copy_value(detail)
        status_lower = str(status).lower()
        if status_lower in _PASS_STATUSES:
            pass_rows.append((label, detail_clean or ""))
        else:
            fail_rows.append((label, detail_clean or "-"))

    lines: List[str] = []

    if pass_rows:
        lines.append("✅ Passed")
        for label, detail in pass_rows:
            if detail and detail != "-":
                short_detail = _shorten_summary_detail(detail)
                if short_detail:
                    lines.append(f"✅ | {label} - {short_detail}")
                else:
                    lines.append(f"✅ | {label}")
            else:
                lines.append(f"✅ | {label}")

    if fail_rows:
        if lines:
            lines.append("")
        lines.append("❌ Failed")
        for label, detail in fail_rows:
            lines.append(f"❌ | {label}")
            if detail and detail != "-":
                short_detail = _shorten_summary_detail(detail)
                if short_detail:
                    lines.append(f"Key Issues: {short_detail}")

    return lines


def _build_qa_summary_table(results: Dict[str, Any]) -> List[str]:
    return []


def _print_qa_comprehensive_summary(results: Dict[str, Any]) -> None:
    rows = _collect_qa_summary_rows(results)
    if not rows:
        return

    items_map = {
        "Title & Meta": "Title, Meta Description, Author",
        "Headings": "H1 Presence, Hierarchy",
        "Internal Links": "Link Health, Contextual Coverage",
        "Schema": "Structured Data Blocks",
        "Image Optimization": "Alt Text, Visible Assets",
        "Responsiveness": "Viewport Meta, Breakpoints",
        "FAQ": "FAQ Headings, FAQPage schema",
        "Indexability": "Robots Directives, X-Robots-Tag",
        "Canonical": "Canonical Tag",
    }

    def _status_colors(status: str) -> str:
        status_lower = status.lower()
        if status_lower == "pass":
            return "bold white on green"
        if status_lower == "fail":
            return "bold white on red"
        return "bold white on dark_orange3"

    if _rich_enabled():
        table = Table(title="QA Comprehensive Summary", box=box.ROUNDED if box else None, header_style="bold cyan")
        table.add_column("Category", style="bold")
        table.add_column("Items Checked")
        table.add_column("Result", justify="center")
        table.add_column("Key Issues")

        for area, status, detail in rows:
            items = items_map.get(area, "-")
            key_issues = detail if status.lower() not in _PASS_STATUSES else "-"
            status_style = _status_colors(status)
            table.add_row(area, items, f"[{status_style}]{status}[/{status_style}]", key_issues)

        _RICH_CONSOLE.print(table)
        return

    print("\nQA Comprehensive Summary")
    header = f"{'Category':20} | {'Items Checked':30} | {'Result':10} | Key Issues"
    print(header)
    print("-" * len(header))
    for area, status, detail in rows:
        items = items_map.get(area, "-")
        key_issues = detail if status.lower() not in _PASS_STATUSES else "-"
        print(f"{area:20} | {items:30} | {status:10} | {key_issues}")





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

    add("# SEO Audit Report")
    add("")
    add(f"**URL:** {url}")
    fetched = results.get('_fetched_at')
    if fetched:
        add(f"**Fetched:** {fetched}")

    qa_comment_lines = _build_qa_comment_lines(results)
    if qa_comment_lines:
        add("")
        add("## QA Comment")
        lines.extend(qa_comment_lines)

    return "\n".join(lines)



def print_results_as_text(results: Dict[str, Any], url: str) -> None:
    report = format_results_as_text(results, url)
    print("\n## Copy-Friendly Audit Report\n")
    print(report)

def print_results_as_tables(results: Dict[str, Any], url: str):
    _print_qa_comprehensive_summary(results)

    # Per-section scores (percent)
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
    other_levels = ", ".join(
        f"{tag.upper()}({count})"
        for tag, count in sorted((faq.get('non_h3_counts') or {}).items())
        if count > 0
    ) or "-"
    _print_table(
        "FAQ",
        ["Detected", "H3 Count", "Other Levels", "Schema FAQPage", "Status", "Message"],
        [[
            "yes" if faq.get('faq_detected') else "no",
            faq.get('h3_count', 0),
            other_levels,
            "yes" if sc.get('faqpage_found') else "no",
            faq.get('status', ''),
            _truncate(faq.get('message', '')),
        ]],
    )
    non_h3_examples = faq.get('non_h3_examples') or []
    if non_h3_examples:
        _print_table(
            "FAQ Non-H3 Examples",
            ["Heading Text"],
            [[_truncate(text, 120)] for text in non_h3_examples[:10]],
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
    content_total = int(im.get('content_image_count') or 0)
    visible_total = int(im.get('total_images') or 0)
    issue_refs: Set[str] = set()
    for ref in (im.get('content_missing_alt') or []):
        if isinstance(ref, str) and ref:
            issue_refs.add(ref)
    for ref in (im.get('content_missing_title') or []):
        if isinstance(ref, str) and ref:
            issue_refs.add(ref)
    issue_count = len(issue_refs)
    if content_total == 0:
        total_found = int(im.get('total_found') or 0)
        hidden = int(im.get('skipped_hidden') or 0)
        img_percent = 100.0 if visible_total == 0 and total_found > 0 and hidden == total_found else 0.0
    else:
        img_percent = 100.0 * (1.0 - issue_count / float(content_total))
    filtered = int(im.get('decorative_images_filtered') or 0)
    message = _truncate(im.get('message', ''))
    if filtered:
        extra_note = f"Filtered {filtered} decorative image(s)."
        message = f"{message} {extra_note}".strip()
    _print_table(
        "Images",
        ["Content", "Visible", "Status", "Percent", "Message"],
        [[content_total, visible_total, im.get('status', ''), f"{img_percent:.0f}%", message]],
    )
    # Combine all alt-related issues into a single table
    issues_rows: List[List[str]] = []
    missing_with_title = set()
    for src in (im.get('missing_alt_with_title') or [])[:20]:
        if isinstance(src, str):
            missing_with_title.add(src)
            issues_rows.append(["Missing alt (has title)", _truncate(src, 120)])
    for src in (im.get('content_missing_alt') or [])[:20]:
        if isinstance(src, str) and src not in missing_with_title:
            issues_rows.append(["Missing alt", _truncate(src, 120)])
    for src in (im.get('content_missing_title') or [])[:20]:
        if isinstance(src, str):
            issues_rows.append(["Missing title", _truncate(src, 120)])
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
        pass




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

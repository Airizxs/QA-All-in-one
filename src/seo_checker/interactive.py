"""Interactive helpers for running targeted SEO audits."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:  # pragma: no cover - optional dependency
    Console = None  # type: ignore
    Table = None  # type: ignore
    box = None  # type: ignore

from . import cli

_CONSOLE: Optional[Console] = Console() if Console is not None else None


def _render_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if _CONSOLE:
        table = Table(title=title, header_style="bold cyan", box=box.ROUNDED if box else None)
        for header in headers:
            table.add_column(header)
        if not rows:
            table.add_row(*(["-"] * len(headers)))
        else:
            for row in rows:
                table.add_row(*[str(col) for col in row])
        _CONSOLE.print(table)
        return

    print(f"\n== {title} ==")
    if not rows:
        print("(no data)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, col in enumerate(row):
            val = str(col)
            if len(val) > widths[idx]:
                widths[idx] = len(val)
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    print(header_line)
    print(sep_line)
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")


def _text_prompt(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:  # pragma: no cover - defensive
        return ""


def _print_title_meta(results: Dict[str, Dict]) -> None:
    tm = results.get("title_meta", {})
    rows = []
    for key, label in (("title", "Title"), ("meta_description", "Meta Description"), ("author", "Author")):
        data = tm.get(key, {}) or {}
        rows.append([
            label,
            data.get("status", ""),
            data.get("message") or data.get("content", ""),
        ])
    _render_table("Title & Meta", ["Item", "Status", "Details"], rows)


def _print_headings(results: Dict[str, Dict]) -> None:
    hd = results.get("headings", {})
    rows = [
        ["H1", hd.get("h1_status", ""), hd.get("h1_content", "") or "(none)"],
        ["Hierarchy", hd.get("h_hierarchy", ""), hd.get("h_hierarchy_message", "") or ""],
        ["Levels", "", ",".join(map(str, hd.get("h_tags_found", []) or [])) or "-"]
    ]
    _render_table("Headings", ["Item", "Status", "Details"], rows)


def _print_schema(results: Dict[str, Dict]) -> None:
    sc = results.get("schema", {})
    rows = [[
        "Schema Found",
        "yes" if sc.get("schema_found") else "no",
        ", ".join(sorted({str(t) for t in (sc.get("types") or [])})) or "-",
    ]]
    _render_table("Schema", ["Present", "Details", "Types"], rows)
    urls = sc.get("urls") or []
    if urls:
        _render_table("Schema Script Sources", ["URL"], [[u] for u in urls[:10]])


def _print_faq(results: Dict[str, Dict]) -> None:
    faq = results.get("faq", {})
    other_levels = ", ".join(
        f"{tag.upper()}({count})"
        for tag, count in sorted((faq.get("non_h3_counts") or {}).items())
        if count > 0
    ) or "-"
    rows = [[
        "yes" if faq.get("faq_detected") else "no",
        faq.get("h3_count", 0),
        other_levels,
        faq.get("status", ""),
        faq.get("message", "") or "",
    ]]
    _render_table("FAQ", ["Detected", "H3 Count", "Other Levels", "Status", "Message"], rows)
    non_h3_examples = faq.get("non_h3_examples") or []
    if non_h3_examples:
        _render_table("FAQ Non-H3 Examples", ["Heading"], [[text] for text in non_h3_examples[:10]])


def _print_mobile(results: Dict[str, Dict]) -> None:
    mobile = results.get("mobile_responsiveness", {})
    rows = [[mobile.get("status", ""), mobile.get("message", "")]]
    _render_table("Mobile Responsiveness", ["Status", "Message"], rows)


def _print_breakpoints(results: Dict[str, Dict]) -> None:
    mobile = results.get("mobile_responsiveness", {})
    bp = (mobile or {}).get("breakpoints") or {}
    rows = []
    for label in ("mobile", "tablet", "desktop"):
        info = bp.get("found", {})
        counts = bp.get("counts", {})
        widths = bp.get("widths", {})
        rows.append([
            label.title(),
            "yes" if info.get(label) else "no",
            counts.get(label, 0),
            ", ".join(str(w) for w in widths.get(label, [])[:6]) or "-",
        ])
    _render_table("Responsive Breakpoints", ["Segment", "Found", "Count", "Widths"], rows)


def _print_indexability(results: Dict[str, Dict]) -> None:
    idx = results.get("indexability", {})
    rows = [[
        idx.get("status", ""),
        idx.get("meta_robots", "") or "-",
        idx.get("x_robots_tag", "") or "-",
        idx.get("message", "") or ""
    ]]
    _render_table("Indexability", ["Status", "Meta Robots", "X-Robots-Tag", "Message"], rows)


def _print_locations(results: Dict[str, Dict]) -> None:
    loc = results.get("locations", {})
    rows = [[
        loc.get("status", ""),
        loc.get("count", 0),
        loc.get("message", "") or ""
    ]]
    _render_table("Locations", ["Status", "Count", "Message"], rows)
    addresses = loc.get("addresses") or []
    if addresses:
        _render_table("Addresses", ["Address"], [[addr] for addr in addresses[:20]])


def _print_hero(results: Dict[str, Dict]) -> None:
    hero = results.get("hero_image", {})
    rows = [[
        hero.get("status", ""),
        hero.get("width", "") or "-",
        hero.get("height", "") or "-",
        hero.get("src", "") or "",
        hero.get("message", "") or ""
    ]]
    _render_table("Hero Image", ["Status", "Width", "Height", "Source", "Message"], rows)


def _print_robots(results: Dict[str, Dict]) -> None:
    rob = results.get("robots_sitemaps", {}).get("robots", {})
    rows = [[
        "yes" if rob.get("present") else "no",
        rob.get("url", "") or "",
    ]]
    _render_table("Robots.txt", ["Present", "URL"], rows)


def _print_sitemaps(results: Dict[str, Dict]) -> None:
    rs = results.get("robots_sitemaps", {}).get("sitemaps", {})
    rows = []
    for validated in rs.get("validated", [])[:10]:
        rows.append([
            validated.get("status", ""),
            validated.get("sitemap_url", ""),
            validated.get("message", "") or "",
        ])
    if not rows:
        rows = [[rs.get("status", ""), "-", rs.get("message", "") or "None validated"]]
    _render_table("Sitemaps", ["Status", "URL", "Message"], rows)


def _print_internal_links(results: Dict[str, Dict]) -> None:
    links = results.get("internal_links", {})
    rows = [[
        links.get("status", ""),
        links.get("checked", 0),
        links.get("message", "") or ""
    ]]
    _render_table("Internal Links", ["Status", "Checked", "Message"], rows)
    broken = links.get("broken") or []
    if broken:
        _render_table("Broken Internal Links", ["URL"], [[url] for url in broken[:20]])


def _print_images(results: Dict[str, Dict]) -> None:
    images = results.get("images", {})
    rows = [[
        images.get("status", ""),
        images.get("content_image_count", 0),
        images.get("total_images", 0),
        images.get("message", "") or ""
    ]]
    _render_table("Images", ["Status", "Content", "Visible", "Message"], rows)
    src_rows = []
    missing_with_title = set()
    for src in (images.get("missing_alt_with_title") or [])[:10]:
        src_rows.append(["Missing alt (has title)", src])
        if isinstance(src, str):
            missing_with_title.add(src)
    for src in (images.get("content_missing_alt") or [])[:10]:
        if isinstance(src, str) and src not in missing_with_title:
            src_rows.append(["Missing alt", src])
    for src in (images.get("content_missing_title") or [])[:10]:
        src_rows.append(["Missing title", src])
    if src_rows:
        _render_table("Image Issues", ["Issue", "Reference"], src_rows)


def _print_canonical(results: Dict[str, Dict]) -> None:
    canonical = results.get("canonical_hreflang", {}).get("canonical", {})
    rows = [[
        canonical.get("status", ""),
        canonical.get("url", "") or "",
        canonical.get("message", "") or "",
        canonical.get("multiple", False)
    ]]
    _render_table("Canonical", ["Status", "URL", "Message", "Multiple"], rows)


def _print_hreflang(results: Dict[str, Dict]) -> None:
    hre = results.get("canonical_hreflang", {}).get("hreflang", {})
    entries = hre.get("entries", [])
    rows = [[entry.get("lang", ""), entry.get("url", "")] for entry in entries[:20]] or [["-", "-"]]
    _render_table("Hreflang", ["Lang", "URL"], rows)


SECTION_MAP: Dict[str, Dict[str, Callable[[Dict[str, Dict]], None]]] = {
    "TITLE": {"label": "Title & Meta", "render": _print_title_meta},
    "HEADINGS": {"label": "Headings", "render": _print_headings},
    "SCHEMA": {"label": "Schema", "render": _print_schema},
    "FAQ": {"label": "FAQ", "render": _print_faq},
    "MOBILE": {"label": "Mobile Responsiveness", "render": _print_mobile},
    "BREAKPOINTS": {"label": "Responsive Breakpoints", "render": _print_breakpoints},
    "INDEXABILITY": {"label": "Indexability", "render": _print_indexability},
    "LOCATIONS": {"label": "Locations", "render": _print_locations},
    "HERO": {"label": "Hero Image", "render": _print_hero},
    "ROBOTS": {"label": "Robots.txt", "render": _print_robots},
    "SITEMAPS": {"label": "Sitemaps", "render": _print_sitemaps},
    "INTERNAL": {"label": "Internal Links", "render": _print_internal_links},
    "IMAGES": {"label": "Images", "render": _print_images},
    "CANONICAL": {"label": "Canonical", "render": _print_canonical},
    "HREFLANG": {"label": "Hreflang", "render": _print_hreflang},
}

# Addresses is a subset of locations; reuse renderer but show addresses table only
SECTION_MAP_WITH_ADDRESS = dict(SECTION_MAP)
SECTION_MAP_WITH_ADDRESS["ADDRESSES"] = {
    "label": "Addresses",
    "render": lambda res: _render_table(
        "Addresses", ["Address"], [[addr] for addr in (res.get("locations", {}).get("addresses") or [])[:20]]
    ),
}


def _capture_screenshot_bundle(url: str) -> None:
    """Capture SEO summary panels and a full-page screenshot via Puppeteer."""
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "tools" / "capture_sections.js"
    if not script_path.exists():
        print("Screenshot script not found at tools/capture_sections.js. Install it to enable captures.\n")
        return

    print("Capturing screenshots (SEO panels + full page). This may take a moment...\n")
    ok, message = cli._run_puppeteer_capture(str(script_path), url, "logs/screenshots", 1000)
    if ok:
        print(f"{message}\n")
    else:
        print(f"Puppeteer capture failed: {message}\n")


def run_audit_flow() -> None:
    url = _text_prompt("Enter the URL to audit (leave blank to cancel): ").strip()
    if not url:
        print("No URL provided. Returning to menu.\n")
        return

    print("\nSelect which checks to run:")
    print("  [0] Cancel")
    print("  [1] Run full SEO audit")
    keys = list(SECTION_MAP_WITH_ADDRESS.keys())
    for idx, key in enumerate(keys, start=2):
        print(f"  [{idx}] {SECTION_MAP_WITH_ADDRESS[key]['label']}")
    selection = _text_prompt("Enter choice (single number or comma-separated): ").strip()
    if not selection:
        selection = "1"

    if selection == "0":
        print("Cancelled.\n")
        return

    try:
        choices = {int(part.strip()) for part in selection.split(',') if part.strip()}
    except ValueError:
        print("Invalid selection. Please enter numbers like '1' or '2,5'.\n")
        return

    results = cli.run_all_checks(url, quiet=True)
    results['_score_summary'] = cli.compute_score(results)
    results['_section_scores'] = cli.compute_section_scores(results)

    if 1 in choices or not choices:
        cli.print_results_as_tables(results, url)
        cli.print_results_as_text(results, url)
        _capture_screenshot_bundle(url)
        return

    max_choice = len(keys) + 1
    chosen_sections = []
    for choice in sorted(choices):
        if choice < 2 or choice > max_choice:
            print(f"Skipping invalid option {choice}.")
            continue
        key = keys[choice - 2]
        chosen_sections.append(key)
        SECTION_MAP_WITH_ADDRESS[key]['render'](results)

    cli._print_recommendations(results)
    _capture_screenshot_bundle(url)

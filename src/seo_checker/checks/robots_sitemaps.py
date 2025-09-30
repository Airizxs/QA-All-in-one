import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from typing import Dict, List
import requests
from ..utils.fetch import build_session

from ..utils.fetch import fetch_html, DEFAULT_HEADERS


def _validate_sitemap(sitemap_url: str, timeout: int, use_scraperapi: bool) -> Dict:
    try:
        # Use a session with default headers/proxy/SSL settings
        session = build_session(DEFAULT_HEADERS)
        r = session.get(sitemap_url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            try:
                ET.fromstring(r.content)
                return {"status": "pass", "message": "Valid sitemap XML", "sitemap_url": sitemap_url}
            except ET.ParseError:
                return {"status": "fail", "message": "Invalid sitemap XML", "sitemap_url": sitemap_url}
    except requests.RequestException:
        pass
    return {}


def check_robots_and_sitemaps(url: str, timeout: int = 20, use_scraperapi: bool = False) -> Dict:
    results: Dict = {
        "robots": {"present": False, "url": None},
        "sitemaps": {"discovered": [], "validated": [], "status": "fail"},
    }

    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(origin, "/robots.txt")

    robots_txt = fetch_html(robots_url, timeout=timeout, use_scraperapi=use_scraperapi)
    if robots_txt:
        results["robots"]["present"] = True
        results["robots"]["url"] = robots_url
        # Collect Sitemap directives
        sitemap_lines: List[str] = [
            line.split(":", 1)[1].strip()
            for line in robots_txt.splitlines()
            if line.lower().startswith("sitemap:") and ":" in line
        ]
    else:
        sitemap_lines = []

    # Try common default locations too
    common = [urljoin(origin, "/sitemap.xml"), urljoin(origin, "/sitemap_index.xml")]
    discovered = list(dict.fromkeys([*sitemap_lines, *common]))
    results["sitemaps"]["discovered"] = discovered

    validated = []
    for sm in discovered:
        outcome = _validate_sitemap(sm, timeout=timeout, use_scraperapi=use_scraperapi)
        if outcome:
            validated.append(outcome)

    results["sitemaps"]["validated"] = validated
    results["sitemaps"]["status"] = "pass" if any(v.get("status") == "pass" for v in validated) else (
        "warning" if validated else "fail"
    )
    return results

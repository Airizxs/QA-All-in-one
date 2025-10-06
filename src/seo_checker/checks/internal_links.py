from typing import Dict, List, Set, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.fetch import build_session, DEFAULT_HEADERS


def _is_tel_link(href: str) -> bool:
    return href.strip().lower().startswith("tel:")


def is_internal(base_url: str, href: str) -> bool:
    href_url = urlparse(href)
    base_url_parsed = urlparse(base_url)
    if not href_url.netloc:
        return True
    return href_url.netloc == base_url_parsed.netloc


def _normalize_link(base_url: str, href: str) -> str:
    full = urljoin(base_url, href)
    parts = urlparse(full)
    # Drop fragments to avoid duplicate checks of the same resource
    return parts._replace(fragment="").geturl()


def _check_one(session: requests.Session, link: str, timeout: int) -> Tuple[str, bool, str]:
    try:
        # Try HEAD first to avoid downloading bodies
        r = session.head(link, timeout=timeout, allow_redirects=True)
        status = r.status_code
        r.close()
        if status == 405 or status < 200 or status >= 400:
            # Fallback to lightweight GET if HEAD not allowed or error-ish
            rg = session.get(link, timeout=timeout, allow_redirects=True, stream=True)
            status = rg.status_code
            rg.close()
        return link, (200 <= status < 400), f"{status}"
    except requests.RequestException as e:
        return link, False, "request failed"


def check_internal_links(html: str, base_url: str, timeout: int = 20, max_links: int = 25, concurrency: int = 8) -> Dict:
    soup = BeautifulSoup(html, 'html.parser')
    internal_links: Set[str] = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if _is_tel_link(href):
            continue
        full = _normalize_link(base_url, href)
        if is_internal(base_url, full):
            internal_links.add(full)

    links_to_check = list(internal_links)[:max_links]
    broken: List[str] = []

    session = build_session(DEFAULT_HEADERS)
    # Parallelize link checks for speed
    if links_to_check:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = [executor.submit(_check_one, session, link, timeout) for link in links_to_check]
            for fut in as_completed(futures):
                link, ok, note = fut.result()
                if not ok:
                    broken.append(f"{link} ({note})")

    # Count contextual links within main content (heuristic): anchors inside <main>, or in <p>/<li> not within header/nav/footer/aside
    def is_in_context(a_tag) -> bool:
        # Exclude common chrome
        for parent in a_tag.parents:
            if parent.name in ("header", "nav", "footer", "aside"):
                return False
        if a_tag.find_parent("main") is not None:
            return True
        parent = a_tag.find_parent(["p", "li", "article", "section"])
        return parent is not None

    contextual_links = []
    for a in soup.find_all('a', href=True):
        if _is_tel_link(a['href']):
            continue
        normalized = _normalize_link(base_url, a['href'])
        if is_internal(base_url, normalized) and is_in_context(a):
            contextual_links.append(normalized)
    contextual_count = len(set(contextual_links))

    status = "pass" if links_to_check and not broken and contextual_count >= 2 else (
        "warning" if links_to_check and contextual_count > 0 else "fail"
    )

    if not links_to_check:
        message = "No internal links found."
    elif broken:
        preview = ", ".join(broken[:3])
        if len(broken) > 3:
            preview += f", +{len(broken) - 3} more"
        message = f"{len(broken)} broken link(s) found. Investigate: {preview}."
    elif contextual_count < 2:
        message = f"Only {contextual_count} contextual link(s); need 2+."
    else:
        message = (
            f"Checked {len(links_to_check)} internal links; all working. Contextual links: {contextual_count}."
        )
    return {
        "total_internal": len(internal_links),
        "checked": len(links_to_check),
        "broken": broken,
        "contextual_links": contextual_count,
        "status": status,
        "message": message,
    }

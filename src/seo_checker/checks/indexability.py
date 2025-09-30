from typing import Dict, Optional
from bs4 import BeautifulSoup
import requests

from ..utils.fetch import build_session, DEFAULT_HEADERS


def _parse_robots_directives(value: str) -> Dict[str, bool]:
    directives = {k.strip().lower(): True for k in (value or "").split(',')}
    return {k: True for k in directives}


def check_indexability(url: str, soup: BeautifulSoup, timeout: int = 10) -> Dict:
    """Check meta robots and X-Robots-Tag to ensure page is indexable."""
    result = {
        "meta_robots": None,
        "x_robots_tag": None,
        "status": "pass",
        "message": "",
    }

    # Meta robots in HTML
    meta = soup.find('meta', attrs={'name': 'robots'})
    if meta and meta.get('content'):
        val = meta['content'].strip()
        result["meta_robots"] = val
        d = _parse_robots_directives(val)
        if 'noindex' in d:
            result['status'] = 'fail'
            result['message'] = (result['message'] + " ").strip() + 'Meta robots contains noindex.'

    # X-Robots-Tag header (best-effort)
    try:
        session = build_session(DEFAULT_HEADERS)
        r = session.head(url, timeout=timeout, allow_redirects=True)
        header_val: Optional[str] = r.headers.get('X-Robots-Tag') or r.headers.get('x-robots-tag')
        if header_val:
            result['x_robots_tag'] = header_val
            d2 = _parse_robots_directives(header_val)
            if 'noindex' in d2:
                result['status'] = 'fail'
                result['message'] = (result['message'] + " ").strip() + 'X-Robots-Tag contains noindex.'
    except requests.RequestException:
        # ignore header failures
        pass

    if result['status'] == 'pass' and not result['message']:
        result['message'] = 'Indexable (no noindex directives found).'
    return result

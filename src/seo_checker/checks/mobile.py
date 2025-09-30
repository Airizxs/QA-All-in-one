import re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from ..utils.fetch import build_session

# Use browser-like headers to reduce 403s during checks
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def _collect_css(soup: BeautifulSoup, base_url: str, session: requests.Session, *, max_files: int = 10, max_bytes: int = 200_000) -> str:
    css_chunks = []
    # Inline <style>
    for st in soup.find_all('style'):
        text = st.string or st.get_text() or ''
        if text:
            css_chunks.append(text)
    # Linked stylesheets
    links = []
    for ln in soup.find_all('link'):
        rel = (ln.get('rel') or [])
        rels = {r.lower() for r in (rel if isinstance(rel, list) else [rel]) if r}
        as_attr = (ln.get('as') or '').lower()
        if 'stylesheet' in rels or ('preload' in rels and as_attr == 'style'):
            href = ln.get('href')
            if href:
                links.append(urljoin(base_url, href))
    # Dedup and cap
    seen = set()
    uniq_links = []
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq_links.append(u)
    for href in uniq_links[:max_files]:
        try:
            r = session.get(href, timeout=10, allow_redirects=True)
            r.raise_for_status()
            content = r.text
            if len(content) > max_bytes:
                content = content[:max_bytes]
            css_chunks.append(content)
        except requests.RequestException:
            continue
    return "\n".join(css_chunks)


def _analyze_breakpoints(css_text: str) -> dict:
    # Find media queries and extract pixel widths
    # Matches (max-width: 768px) or (min-width:1024px) etc.
    widths = []
    for m in re.finditer(r"\(\s*(?:max|min)-(?:device-)?width\s*:\s*(\d+)px\s*\)", css_text, re.I):
        try:
            widths.append(int(m.group(1)))
        except ValueError:
            continue
    # Categorize
    buckets = {
        'mobile': [w for w in widths if w <= 600],
        'tablet': [w for w in widths if 601 <= w <= 1024],
        'desktop': [w for w in widths if w >= 1025],
    }
    return {
        'found': {k: (len(v) > 0) for k, v in buckets.items()},
        'counts': {k: len(v) for k, v in buckets.items()},
        'widths': {k: sorted(set(v))[:10] for k, v in buckets.items()},
    }


def check_mobile_responsiveness(url: str) -> dict:
    """
    A basic check for mobile responsiveness by looking for the viewport meta tag.
    A more advanced check would require a headless browser or an API.

    Args:
        url (str): The URL of the website to check.

    Returns:
        dict: A dictionary with the results of the check.
    """
    results = {
        'viewport_meta_tag': {'found': False, 'content': None},
        'status': 'fail'
    }

    try:
        session = build_session(DEFAULT_HEADERS)
        response = session.get(url, timeout=10, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'message': f"Failed to access the URL for mobile check: {e}"}

    # Look for the viewport meta tag
    viewport_tag = soup.find('meta', attrs={'name': 'viewport'})
    
    if viewport_tag and 'content' in viewport_tag.attrs:
        results['viewport_meta_tag']['found'] = True
        results['viewport_meta_tag']['content'] = viewport_tag['content']
        
        # Simple check for common viewport settings
        if 'width=device-width' in viewport_tag['content'] and 'initial-scale=1' in viewport_tag['content']:
            results['status'] = 'pass'
            results['message'] = 'Viewport meta tag with common settings found.'
        else:
            results['status'] = 'warning'
            results['message'] = 'Viewport tag found, but content is not a standard mobile configuration.'
    else:
        results['status'] = 'fail'
        results['message'] = 'No viewport meta tag found, which is essential for mobile responsiveness.'

    # Analyze breakpoints from CSS (inline + linked)
    try:
        css_text = _collect_css(soup, url, session)
        bp = _analyze_breakpoints(css_text)
        results['breakpoints'] = bp
        # Upgrade/downgrade status based on breakpoint evidence
        any_bp = any(bp['found'].values())
        if results['status'] == 'pass' and not any_bp:
            # Viewport present but no media queries — still acceptable, keep pass
            pass
        elif results['status'] == 'warning' and any_bp:
            results['message'] += ' Responsive media queries detected.'
        elif results['status'] == 'fail' and any_bp:
            results['status'] = 'warning'
            results['message'] = 'No viewport tag, but responsive media queries detected.'
    except Exception:
        # Non-fatal; ignore CSS analysis errors
        pass

    return results

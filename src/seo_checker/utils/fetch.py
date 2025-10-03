import os
import requests
from typing import Optional, Dict

# Default browser-like headers to reduce 403s
DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def build_session(headers: Optional[Dict[str, str]] = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(headers or DEFAULT_HEADERS)
    # Proxies from environment (HTTP(S)_PROXY)
    proxies: Dict[str, str] = {}
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    if proxies:
        session.proxies.update(proxies)
    # SSL verification control: REQUESTS_CA_BUNDLE or disable via SEO_CHECKER_INSECURE
    insecure = os.environ.get("SEO_CHECKER_INSECURE") == "1"
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if insecure:
        session.verify = False
    elif ca_bundle:
        session.verify = ca_bundle
    return session


def fetch_html(url: str, timeout: int = 20, use_scraperapi: bool = False, scraperapi_key: Optional[str] = None) -> Optional[str]:
    """Fetch HTML using ScraperAPI if requested and configured, else direct session with strong headers."""
    scraperapi_key = scraperapi_key or os.environ.get("SCRAPERAPI_KEY")
    try:
        if use_scraperapi and scraperapi_key:
            params = {"api_key": scraperapi_key, "url": url}
            r = requests.get("http://api.scraperapi.com/", params=params, timeout=max(timeout, 30))
            r.raise_for_status()
            return r.text

        session = build_session()
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 403:
            # Retry with a different UA string
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
                )
            })
            r = session.get(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except requests.RequestException:
        return None

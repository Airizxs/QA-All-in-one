from typing import Dict, List
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


def check_canonical_and_hreflang(soup: BeautifulSoup, base_url: str) -> Dict:
    results: Dict = {
        "canonical": {
            "found": False,
            "url": None,
            "multiple": False,
            "status": "fail",
            "message": "Canonical not found",
        },
        "hreflang": {
            "entries": [],
            "duplicates": [],
            "invalid": [],
            "status": "pass",
            "message": "",
        },
    }

    # Canonical
    canon_links = soup.find_all('link', rel=lambda v: v and 'canonical' in v)
    if canon_links:
        results["canonical"]["found"] = True
        results["canonical"]["multiple"] = len(canon_links) > 1
        hrefs = [l.get('href') for l in canon_links if l.get('href')]
        if hrefs:
            canonical_url = urljoin(base_url, hrefs[0])
            results["canonical"]["url"] = canonical_url
            results["canonical"]["status"] = "warning" if len(canon_links) > 1 else "pass"
            results["canonical"]["message"] = (
                "Multiple canonical tags present" if len(canon_links) > 1 else "Canonical present"
            )
        else:
            results["canonical"]["status"] = "fail"
            results["canonical"]["message"] = "Canonical tag missing href"

    # Hreflang
    hreflangs = soup.find_all('link', rel=lambda v: v and 'alternate' in v, hreflang=True)
    seen_langs: List[str] = []
    duplicates: List[str] = []
    invalid: List[str] = []
    entries = []
    for tag in hreflangs:
        lang = tag.get('hreflang', '').strip()
        href = tag.get('href')
        if not href or not lang:
            continue
        abs_url = urljoin(base_url, href)
        entries.append({"lang": lang, "url": abs_url})
        # Basic validation: language code pattern like en or en-US or x-default
        if lang.lower() != 'x-default' and not (len(lang) in (2, 5) and (len(lang) == 2 or '-' in lang)):
            invalid.append(lang)
        if lang in seen_langs:
            duplicates.append(lang)
        else:
            seen_langs.append(lang)

    status = "pass"
    msg_parts = []
    if duplicates:
        status = "warning"
        msg_parts.append(f"Duplicate hreflang entries: {sorted(set(duplicates))}")
    if invalid:
        status = "warning"
        msg_parts.append(f"Invalid hreflang codes: {sorted(set(invalid))}")
    results["hreflang"]["entries"] = entries
    results["hreflang"]["duplicates"] = sorted(set(duplicates))
    results["hreflang"]["invalid"] = sorted(set(invalid))
    results["hreflang"]["status"] = status
    results["hreflang"]["message"] = "; ".join(msg_parts) if msg_parts else "Hreflang tags look OK"

    return results


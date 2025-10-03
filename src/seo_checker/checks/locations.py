import json
import re
from typing import Dict, List, Any, Set, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote


US_STATE_RE = r"AL|AK|AS|AZ|AR|CA|CO|CT|DC|DE|FL|GA|GU|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MP|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|PR|RI|SC|SD|TN|TX|UM|UT|VA|VI|VT|WA|WI|WV|WY"
US_STATE_NAMES = [
    'Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming','District of Columbia'
]
STATE_NAMES_RE = r"(?:" + "|".join(re.escape(n) for n in US_STATE_NAMES) + r")"
STREET_TYPES = r"St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Cir|Circle|Pkwy|Parkway|Hwy|Highway|Pl|Place|Terr|Terrace|Sq|Square"
DIR_WORD = r"(?:N|S|E|W|North|South|East|West)\.?"
# More tolerant US address pattern:
#  - number + optional directional + multi-word street + required street type
#  - optional directional suffix
#  - optional unit (Suite/Ste./Apt/Unit/# ...)
#  - city, state (2-letter or spelled out), ZIP (5 or 9)
US_ADDRESS_RE = re.compile(
    rf"\b("  # capture full street part
    rf"\d+\s+"  # street number
    rf"(?:{DIR_WORD}\s+)?"  # optional directional before street name
    rf"(?:[A-Za-z0-9\.'-]+\s+)+"  # one or more words/numbers in street name
    rf"(?:{STREET_TYPES})\b"  # street type
    rf"(?:\s+{DIR_WORD})?"  # optional directional suffix
    rf")\s*,?\s*"
    rf"(?:,?\s*(?:Ste\.?|Suite|Unit|Apt|#)\s*[\w-]+)?\s*,?\s*"  # optional unit
    rf"([A-Za-z\.'\- ]+),\s*"  # city
    rf"((?:{US_STATE_RE})|(?:{STATE_NAMES_RE}))\s*"  # state (2-letter or spelled-out)
    rf"(\d{{5}}(?:-\d{{4}})?)\b",  # ZIP
    re.I,
)


def _flatten_json(obj: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        items.append(obj)
        for v in obj.values():
            items.extend(_flatten_json(v))
    elif isinstance(obj, list):
        for it in obj:
            items.extend(_flatten_json(it))
    return items


def _build_address_str(addr: Dict[str, Any]) -> str:
    parts: List[str] = []
    if addr.get('streetAddress'):
        parts.append(str(addr['streetAddress']))
    city_line: List[str] = []
    if addr.get('addressLocality'):
        city_line.append(str(addr['addressLocality']))
    if addr.get('addressRegion'):
        city_line.append(str(addr['addressRegion']))
    if addr.get('postalCode'):
        city_line.append(str(addr['postalCode']))
    if city_line:
        parts.append(", ".join(city_line[:-1]) + (" " + city_line[-1] if len(city_line) >= 1 else ""))
    if addr.get('addressCountry'):
        parts.append(str(addr['addressCountry']))
    return ", ".join([p for p in parts if p])


def _extract_from_jsonld(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    for tag in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(tag.string or "{}")
        except Exception:
            continue
        for node in _flatten_json(data):
            t = node.get('@type')
            types = ([t] if isinstance(t, str) else (t or [])) if t else []
            # PostalAddress directly
            if any(str(x).lower() == 'postaladdress' for x in types) or {
                'streetAddress', 'addressLocality', 'addressRegion', 'postalCode'
            } & set(node.keys()):
                out.append(_build_address_str(node))
            # address nested
            addr = node.get('address')
            if isinstance(addr, dict):
                out.append(_build_address_str(addr))
            elif isinstance(addr, list):
                for a in addr:
                    if isinstance(a, dict):
                        out.append(_build_address_str(a))
    return [s for s in {s.strip().strip(',') for s in out if s and s.strip()}]


def _extract_from_microdata(soup: BeautifulSoup) -> List[str]:
    out: List[str] = []
    # Simple microdata heuristic: elements with itemtype containing PostalAddress
    for el in soup.find_all(attrs={'itemtype': True}):
        it = str(el.get('itemtype') or '').lower()
        if 'postaladdress' in it:
            addr_dict: Dict[str, str] = {}
            for prop in ('streetAddress', 'addressLocality', 'addressRegion', 'postalCode', 'addressCountry'):
                tag = el.find(attrs={'itemprop': prop})
                if tag:
                    addr_dict[prop] = tag.get_text(" ", strip=True) if tag else ''
            out.append(_build_address_str(addr_dict))
    return [s for s in {s.strip().strip(',') for s in out if s and s.strip()}]


def _extract_from_text(soup: BeautifulSoup) -> List[str]:
    # Look in footer, contact, location sections
    candidates: List[str] = []
    selectors = [
        {'id': re.compile(r'(footer|contact|location)', re.I)},
        {'class': re.compile(r'(footer|contact|location|office)', re.I)},
    ]
    containers: List[Any] = []
    for sel in selectors:
        containers.extend(soup.find_all(True, attrs=sel))
    # Also include semantic <footer> elements
    containers.extend(soup.find_all('footer'))
    containers = list(dict.fromkeys(containers))
    text_blobs: List[str] = []
    for c in containers[:20]:  # limit
        text_blobs.append(c.get_text("\n", strip=True))
    # Always include the whole page text as a fallback to avoid misses
    text_blobs.append(soup.get_text("\n", strip=True))
    found: Set[str] = set()
    for blob in text_blobs:
        for m in US_ADDRESS_RE.finditer(blob):
            street, city, state, zipc = m.groups()
            # Normalize state to 2-letter if spelled out
            st = state
            if len(state) > 2:
                try:
                    idx = next(i for i, name in enumerate(US_STATE_NAMES) if name.lower() == state.lower())
                    # Mapping list order to abbreviations via parallel list
                    ABBRS = [
                        'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'
                    ]
                    st = ABBRS[idx]
                except StopIteration:
                    st = state
            addr = f"{street}, {city}, {st} {zipc}"
            found.add(addr)
    return sorted(found)


def _extract_from_google_maps(soup: BeautifulSoup) -> List[str]:
    out: Set[str] = set()
    # Look in iframes and links that point to Google Maps
    candidates = []
    for tag in soup.find_all(['iframe', 'a']):
        url = tag.get('src') or tag.get('href')
        if not url:
            continue
        if any(host in url for host in ['google.com/maps', 'goo.gl/maps', 'maps.app.goo.gl']):
            candidates.append(url)
    for url in candidates[:20]:
        try:
            u = urlparse(url)
            qs = parse_qs(u.query)
            parts: List[str] = []
            # Prefer 'q' parameter if present
            if 'q' in qs and qs['q']:
                parts.append(unquote(qs['q'][0]))
            # Some URLs encode address in the path, e.g., /place/<name>/@...
            path = unquote(u.path or '')
            parts.append(path.replace('/place/', '').replace('+', ' '))
            blob = " \n ".join(parts)
            for m in US_ADDRESS_RE.finditer(blob):
                street, city, state, zipc = m.groups()
                out.add(f"{street}, {city}, {state} {zipc}")
        except Exception:
            continue
    return sorted(out)


def check_locations(soup: BeautifulSoup) -> Dict[str, Any]:
    addresses: List[str] = []
    try:
        addresses.extend(_extract_from_jsonld(soup))
    except Exception:
        pass
    try:
        addresses.extend(_extract_from_microdata(soup))
    except Exception:
        pass
    try:
        addresses.extend(_extract_from_text(soup))
    except Exception:
        pass
    try:
        addresses.extend(_extract_from_google_maps(soup))
    except Exception:
        pass

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen: Set[str] = set()
    for a in addresses:
        key = re.sub(r"\s+", " ", a.strip().lower().strip(','))
        if key and key not in seen:
            seen.add(key)
            deduped.append(a)

    status = 'pass' if deduped else 'fail'
    return {
        'status': status,
        'count': len(deduped),
        'addresses': deduped[:50],
        'message': (f"Found {len(deduped)} address(es)." if deduped else 'No addresses detected on page.'),
    }

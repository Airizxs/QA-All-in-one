from typing import Dict
from bs4 import BeautifulSoup


def check_faq(soup: BeautifulSoup) -> Dict:
    """Heuristic FAQ check: looks for H3s in FAQ-like sections (class/id contains 'faq')."""
    faq_sections = []
    # Sections with class or id containing 'faq'
    for el in soup.find_all(True, attrs={"class": True}):
        classes = " ".join(el.get("class") or [])
        if 'faq' in classes.lower():
            faq_sections.append(el)
    for el in soup.find_all(True, attrs={"id": True}):
        if 'faq' in (el.get('id') or '').lower():
            faq_sections.append(el)

    faq_sections = list(dict.fromkeys(faq_sections))

    h3_count = 0
    if faq_sections:
        for sec in faq_sections:
            h3_count += len(sec.find_all('h3'))
    else:
        # fallback: any H3s on page
        h3_count = len(soup.find_all('h3'))

    status = 'pass' if h3_count > 0 else 'fail'
    return {
        'h3_count': h3_count,
        'status': status,
        'message': ('H3 FAQ headings found.' if h3_count > 0 else 'No H3 headings detected for FAQ.'),
    }


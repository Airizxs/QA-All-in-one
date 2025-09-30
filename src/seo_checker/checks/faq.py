from typing import Dict, List
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
    h5_count = 0
    h5_texts: List[str] = []

    def _collect_counts(node: BeautifulSoup):
        nonlocal h3_count, h5_count
        h3_count += len(node.find_all('h3'))
        h5_nodes = node.find_all('h5')
        h5_count += len(h5_nodes)
        for heading in h5_nodes[:10]:
            text = heading.get_text(strip=True)
            if text:
                h5_texts.append(text)

    if faq_sections:
        for sec in faq_sections:
            _collect_counts(sec)
    else:
        _collect_counts(soup)

    if h3_count == 0:
        status = 'fail'
        message = 'No H3 headings detected for FAQ.'
    elif h5_count > 0:
        status = 'warning'
        message = f'Found {h5_count} FAQ heading(s) using H5; use H3 for FAQs.'
    else:
        status = 'pass'
        message = 'H3 FAQ headings found.'

    return {
        'h3_count': h3_count,
        'h5_count': h5_count,
        'h5_examples': h5_texts,
        'status': status,
        'message': message,
    }

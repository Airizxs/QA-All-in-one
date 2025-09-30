from typing import List, Dict, Any
from bs4 import BeautifulSoup
import json

def check_schema(soup: BeautifulSoup) -> dict:
    """
    Checks for the presence of JSON-LD schema markup.

    Args:
        soup (BeautifulSoup): The parsed HTML content of the page.

    Returns:
        dict: A dictionary with the results of the schema check.
    """
    results = {
        'schema_found': False,
        'schemas': [],
        'types': [],
        'urls': [],
        'authors': [],
        'faqpage_found': False,
    }

    # Find all <script> tags with the type "application/ld+json"
    schema_tags = soup.find_all('script', type='application/ld+json')

    if schema_tags:
        results['schema_found'] = True
        for tag in schema_tags:
            src_attr = tag.get('src')
            if src_attr:
                results['urls'].append(src_attr)
            try:
                # Attempt to parse the JSON content
                schema_content = json.loads(tag.string)
            except json.JSONDecodeError:
                results['schemas'].append({'error': 'Invalid JSON in schema script'})
                continue
            results['schemas'].append(schema_content)
            blocks: List[Dict[str, Any]] = schema_content if isinstance(schema_content, list) else [schema_content]
            for b in blocks:
                t = b.get('@type')
                if isinstance(t, list):
                    results['types'].extend([str(x) for x in t])
                elif isinstance(t, str):
                    results['types'].append(t)
                # Capture authors for Article/BlogPosting
                t_lower = str(t).lower() if isinstance(t, str) else None
                if t_lower in ("article", "blogposting"):
                    author = b.get('author')
                    names: List[str] = []
                    if isinstance(author, dict):
                        n = author.get('name')
                        if n:
                            names.append(str(n))
                    elif isinstance(author, list):
                        for it in author:
                            if isinstance(it, dict) and it.get('name'):
                                names.append(str(it['name']))
                            elif isinstance(it, str):
                                names.append(it)
                    elif isinstance(author, str):
                        names.append(author)
                    if names:
                        results['authors'].extend(names)
                # Detect FAQPage type
                if (isinstance(t, str) and t.lower() == 'faqpage') or (
                    isinstance(t, list) and any(str(x).lower() == 'faqpage' for x in t)
                ):
                    results['faqpage_found'] = True
    
    return results

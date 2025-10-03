import os
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify

from seo_checker import run_all_checks, compute_score, compute_section_scores


app = Flask(__name__)


@app.get("/api/health")
def health() -> Any:
    return {"status": "ok"}


@app.post("/api/check")
def api_check() -> Any:
    data = request.get_json(silent=True) or {}
    urls: List[str] = []
    if isinstance(data.get("urls"), list):
        urls = [str(u) for u in data.get("urls") if u]
    if data.get("url"):
        urls.append(str(data.get("url")))
    urls = list(dict.fromkeys(urls))
    if not urls:
        return jsonify({"error": "Provide 'url' or 'urls' in JSON."}), 400

    timeout = int(data.get("timeout") or 20)
    use_scraperapi = bool(data.get("use_scraperapi") or False)
    max_links = int(data.get("max_links") or 25)
    keyword: Optional[str] = data.get("keyword")
    threshold = float(data.get("threshold") or 80.0)

    results: List[Dict[str, Any]] = []
    for u in urls:
        r = run_all_checks(
            u,
            timeout=timeout,
            use_scraperapi=use_scraperapi,
            max_links=max_links,
            quiet=True,
            keyword=keyword,
        )
        if "error" not in r:
            r["_score_summary"] = compute_score(r, threshold=threshold)
            r["_section_scores"] = compute_section_scores(r)
        results.append({"url": u, "results": r})

    return jsonify({"items": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

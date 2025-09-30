# QA all in one

Command-line tool and HTTP API to run comprehensive SEO checks against web pages: title/meta, headings, schema, mobile viewport, robots/sitemaps, internal links, images (alt quality and size), canonical/hreflang, indexability, FAQ, and locations.

## Project Structure

```
.
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── src/
│   └── seo_checker/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── api/
│       │   └── server.py
│       ├── checks/
│       │   ├── canonical.py
│       │   ├── faq.py
│       │   ├── headings.py
│       │   ├── images.py
│       │   ├── indexability.py
│       │   ├── internal_links.py
│       │   ├── locations.py
│       │   ├── mobile.py
│       │   ├── robots_sitemaps.py
│       │   ├── schema.py
│       │   └── title_meta.py
│       └── utils/
│           └── fetch.py
└── tests/
```

The package follows a modern `src/` layout so it can be installed locally or published to PyPI. The CLI and Flask API both reuse the same core modules.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # installs seo-checker in editable mode
```

To upgrade dependencies, adjust `pyproject.toml` and reinstall: `pip install --upgrade -r requirements.txt`.

## Running the CLI

```
seo-checker https://example.com --keyword "target phrase"
```

Alternative module invocation:

```
python -m seo_checker https://example.com --format both --output-json out.json
```

Open the command-center menu (default behaviour):

```
python -m seo_checker
```

Run the CLI directly with arguments:

```
python -m seo_checker --help
python -m seo_checker https://example.com --format both
```

Within the command center choose the SEO audit option to either run the full checklist or target specific areas (title/meta, headings, schema, FAQ, mobile breakpoints, indexability, locations, addresses, hero image, robots/sitemaps, internal links, images, canonical, hreflang).

Key options:

- `--timeout 20` request timeout seconds
- `--max-links 25` internal links to validate
- `--format table|json|both` console output format
- `--threshold 80` pass/fail percentage for exit code
- `--use-scraperapi` use ScraperAPI when `SCRAPERAPI_KEY` is set
- `--history-file` path for JSONL run history (default `.seo_checker_history.jsonl`)

Environment variables:

- `HTTP_PROXY` / `HTTPS_PROXY` HTTP(S) proxy
- `REQUESTS_CA_BUNDLE` custom CA bundle path
- `SEO_CHECKER_INSECURE=1` disable TLS verification (not recommended)
- `SCRAPERAPI_KEY` API key for ScraperAPI (when `--use-scraperapi`)

## Running the HTTP API

Spin up the Flask service locally (useful before deploying to Firebase/Cloud Run):

```
python -m seo_checker.api.server  # http://localhost:8080
```

Or with gunicorn:

```
gunicorn -b :8080 seo_checker.api.server:app
```

Endpoints:

- `GET /api/health` → `{ "status": "ok" }`
- `POST /api/check` with JSON `{ "url": "https://…", "keyword": "…" }`
  - Optional: `urls` (array), `timeout`, `use_scraperapi`, `max_links`, `threshold`

## What It Checks and How

- **Title & Meta**
  - Parses `<title>`, meta description, and optional author meta. Validates presence/length and keyword usage.
- **Headings**
  - Ensures sensible H1 count and heading hierarchy (no skipped levels).
- **Schema (JSON‑LD)**
  - Parses `<script type="application/ld+json">` blocks; collects `@type`; flags FAQPage schema and surfaces external script URLs for review.
- **Mobile Responsiveness**
  - Verifies viewport meta tag and scans inline/linked CSS for responsive breakpoints.
- **Robots & Sitemaps**
  - Requests `/robots.txt`, extracts sitemap URLs, validates reachability and XML health.
- **Internal Links**
  - Normalises internal anchors, checks up to `--max-links` concurrently, counts contextual links.
- **Images**
  - Flags missing and weak alt text, optionally fetches sizes (Content-Length) for up to 20 images.
- **Hero Image**
  - Confirms a hero image is present and reports its captured dimensions (warns only when dimensions are missing).
- **Canonical & Hreflang**
  - Validates canonical URL presence/uniqueness; summarises hreflang entries, duplicates, and invalids.
- **Indexability**
  - Reviews meta robots and X-Robots-Tag headers; reports pass/warn/fail with reasoning.
- **FAQ**
  - Detects FAQ-style heading content and FAQPage schema blocks; warns when FAQ items use `h5` instead of the required `h3` headings.
- **Locations**
  - Extracts addresses from JSON-LD, microdata, visible footer/contact text, and Google Maps embeds.

## Output & Scoring

- Tables per section with status colouring in TTY mode
- Automatically generates colour-coded summary tables and a recommendations list when run in a TTY (powered by `rich`).
- Optional JSON via `--format json` or `--format both`
- CSV export via `--output-csv`
- Composite score aggregates section results (title/meta, headings, schema, mobile, indexability, robots/sitemaps, internal links, images, canonical/hreflang, locations)
- Exit codes: `0` = pass, `1` = below threshold or warnings/failures, `2` = network/fetch failure

## Performance Notes

- Browser-like headers improve fetch reliability; retries 403 responses.
- Internal link checks execute concurrently, preferring HEAD requests.
- Image size sampling is capped (first 20 images) to limit overhead.

## Versioning & Release Workflow

- `main` stays deployable; protect it in GitHub and require PR reviews.
- Feature work happens on `feature/<short-description>` branches.
- Cut release branches as `release/vX.Y` when hardening for production.
- Tag releases from the release branch (`git tag vX.Y.Z`) after QA sign-off, then merge back to `main` and `develop` (if you maintain one).
- Maintain changelog entries per release and bump the version in `pyproject.toml` during the release branch.

## Preparing the GitHub Remote

1. Create a GitHub repository (private or public) under your account/organisation.
2. Add the remote locally: `git remote add origin git@github.com:<org>/<repo>.git`.
3. Ensure `.gitignore` ignores virtualenvs, caches, and the history JSONL file.
4. Commit the reorganised structure: `git add . && git commit -m "chore: bootstrap package layout"`.
5. Push the default branch: `git push -u origin main`.
6. Configure branch protection and required CI (e.g., lint/test workflows) in the repository settings.

## Cloud Run & Firebase Hosting

The provided `Dockerfile` builds the Flask API for Cloud Run and stays compatible with Firebase Hosting rewrites.

Build and deploy:

```
gcloud builds submit --tag gcr.io/PROJECT_ID/seo-checker

gcloud run deploy seo-checker \
  --image gcr.io/PROJECT_ID/seo-checker \
  --platform managed \
  --region REGION \
  --allow-unauthenticated
```

Expose through Firebase Hosting via `firebase.json`:

```
{
  "hosting": {
    "rewrites": [
      { "source": "/api/**", "run": { "serviceId": "seo-checker", "region": "REGION" } }
    ]
  }
}
```

Then deploy Hosting:

```
firebase deploy --only hosting
```

Notes:

- Firebase Functions are Node.js; for Python services prefer Cloud Run + Hosting rewrites.
- The CLI still works behind firewalls; use `--proxy`, `--ca-bundle`, or `--insecure` as needed.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for Cloud Run
ENV PORT=8080

CMD ["gunicorn", "-b", ":8080", "seo_checker.api.server:app", "--workers", "2", "--threads", "4", "--timeout", "120"]

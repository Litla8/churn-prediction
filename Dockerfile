# Churn scoring API.  Build:  docker build -t churn-api .   Run:  docker run --rm -p 8000:8000 churn-api
# The image bundles models/churn_pipeline.joblib and reports/metrics.json, so run
# `python -m churn.pipeline` first. scikit-learn in the image must match the version used to
# train the model (see requirements-api.txt).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_PATH=models/churn_pipeline.joblib \
    CHURN_METRICS_PATH=reports/metrics.json

# libgomp is required by LightGBM in case the selected model is a LightGBM pipeline.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY churn/ churn/
COPY models/churn_pipeline.joblib models/churn_pipeline.joblib
COPY reports/metrics.json reports/metrics.json

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "churn.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

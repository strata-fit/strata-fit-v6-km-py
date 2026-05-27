FROM python:3.11-slim

ARG PKG_NAME="strata_fit_v6_km_py"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PKG_NAME=${PKG_NAME}

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir /app

CMD ["python", "-m", "strata_fit_v6_km_py.container"]

FROM readthedocs/build:ubuntu-22.04-2024.01.29

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libtirpc-dev \
    gcc \
    r-base \
    r-base-dev \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/R-packages \
    && Rscript -e "install.packages('nnTensor', repos='https://cloud.r-project.org', lib='/opt/R-packages')"

ENV R_LIBS=/opt/R-packages

RUN pip install --upgrade pip
RUN pip install uv
RUN uv pip install --system setuptools wheel

WORKDIR /app

COPY imml imml
COPY docs docs
COPY tutorials tutorials
COPY LICENSE .
COPY README.md .
COPY pyproject.toml .
RUN uv pip install --no-cache-dir --system .[docs]


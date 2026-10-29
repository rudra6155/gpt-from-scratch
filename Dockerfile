# CPU image: builds the venv exactly as `make install` does and runs the test suite.
#   make docker        -> docker build -t gpt-scratch .
#   make docker-test   -> docker run --rm gpt-scratch make test
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first so code edits don't invalidate the (large) torch layer.
COPY pyproject.toml README.md Makefile ./
RUN python3 -m venv .venv \
    && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && .venv/bin/pip install numpy matplotlib pytest

COPY . .
RUN make install

CMD ["make", "test"]

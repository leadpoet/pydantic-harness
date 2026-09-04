# Build with ``docker build --platform linux/amd64`` as shown in the README.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /agent

COPY requirements.txt /agent/requirements.txt
COPY experiments/harness_bakeoff/adapters/requirements-pydantic-ai.txt /agent/requirements-pydantic-ai.txt
RUN python -m pip install --no-cache-dir \
    -r /agent/requirements.txt \
    -r /agent/requirements-pydantic-ai.txt

COPY . /agent
COPY agent/run /agent/run
RUN chmod 0555 /agent/run

# The Arena host ignores image ENTRYPOINT and always calls this exact file.
ENTRYPOINT ["/agent/run"]

FROM python:3.12.3-slim

WORKDIR /app

# Install gcc and build dependencies
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

COPY pyproject.toml /app/pyproject.toml

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

CMD ["poetry", "run", "python", "main.py"]
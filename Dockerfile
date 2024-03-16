FROM python:3.11.6
RUN useradd -m agentic
USER agentic
ENV HOME=/home/agentic \
    PATH=/home/agentic/.local/bin:$PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=true
WORKDIR /app
COPY --chown=agentic ./ ./
RUN pip install --upgrade poetry --no-cache-dir && \
    poetry install --with main --no-cache --no-interaction \
    --no-ansi
EXPOSE 8001
CMD ["poetry", "run", "python", "serve.py"]
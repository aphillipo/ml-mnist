FROM python:3-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV INSIDE_DOCKER=1
ENV PYTHONPATH=/app/src

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync

COPY src ./src
COPY .streamlit model_weights.cpu.pth ./

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://0.0.0.0:8080/_stcore/health

CMD ["uv", "run", "streamlit", "run", "/app/src/mnist/ui/streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]

# uv run streamlit run /app/src/mnist/ui/streamlit.py --server.port=8080 --server.address=0.0.0.0
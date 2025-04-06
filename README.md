# ml-mnist

## Install the packages

Make sure you have `uv` installed globally:

```bash
pip3 install uv
```

Then you can install the required packages:

```bash
uv sync
```

## Run the training

```bash
PYTHONPATH=./src uv run python ./src/mnist/model/train.py
```

Note there is already a CPU training available

## Run the project locally

```bash
uv sync
uv run streamlit run /app/src/mnist/ui/streamlit.py --server.port=8080 --server.address=0.0.0.0
```

## Run the project with Docker

```bash
docker-compose up
```

## Run the project on fly.io

```bash
fly deploy
fly posgres create
```

You should get some database connection paramters back.

Set database secrets in your machine (created in fly deploy) for the following:

```env
POSTGRES_DATABASE=mnist
POSTGRES_HOST=supplied above should end in .flycast
POSTGRES_USER=supplied above
POSTGRES_PASSWORD=supplied above
POSTGRES_PORT=5432
```

This will allow the application to conenct to postgres.

NOTE: fly.io has NVIDIA GPUs in their remote builders so you may end up with larger downloads
we force the CPU only version of these packages in pyproject.toml but you may want to experiment
with CUDA deployments in the future.

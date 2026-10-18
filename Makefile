.PHONY: install data eda train test lint api docker
install:
	python -m pip install -r requirements.txt
data:
	python -m src.download_dataset
eda: data
	python -m src.eda
train: data
	python -m src.train
test:
	pytest -q
lint:
	ruff check src tests
api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
docker:
	docker compose up --build

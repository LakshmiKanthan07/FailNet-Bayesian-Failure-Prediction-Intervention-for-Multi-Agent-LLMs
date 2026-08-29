.PHONY: install data train experiment dashboard pipeline clean

install:
	pip install -r requirements.txt

data:
	python -m src.data_ingestion

train:
	python -m src.bn_learning

experiment:
	python -m experiments.exp1_bn_vs_baseline

dashboard:
	streamlit run dashboard/app.py

pipeline:
	python run_pipeline.py

clean:
	rm -rf data/processed/*
	rm -f failnet_model.bif
	find . -type d -name "__pycache__" -exec rm -rf {} +

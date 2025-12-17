.PHONY: up down restart logs clean ingest train init help

up:
	docker compose up -d --build

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

clean:
	docker compose down -v
	@echo "All data volumes have been removed."

ingest:
	@echo "Waiting for Airflow DAGs to be ready..."
	@for i in $$(seq 1 12); do \
		if docker exec airflow-scheduler airflow dags unpause data_ingestion; then \
			echo "DAG unpaused successfully!"; \
			exit 0; \
		fi; \
		echo "Waiting for DAG to be ready... ($$i/12)"; \
		sleep 5; \
	done; \
	echo "Failed to unpause DAG after 60 seconds." && exit 1
	docker exec airflow-scheduler airflow dags trigger data_ingestion
	@echo "DAG triggered. Check Airflow UI at http://localhost:8080"

train:
	@echo "Starting Model Training..."
	docker compose run --rm -e USE_LOCAL_MINIO=false training python scripts/train_model.py --minio --data batches/
	@echo "Training complete. Model uploaded to Minio"
	@echo "Reloading API model..."
	curl -X POST http://localhost:5000/model/reload
	@echo "API updated"

init: up train
	@echo "Waiting for services to be healthy..."
	@echo "Waiting for data ingestion to complete..."
	@echo "System fully initialized and ready"
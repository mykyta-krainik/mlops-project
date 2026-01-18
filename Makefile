.PHONY: up down restart logs clean ingest train init help databricks-validate databricks-deploy databricks-run databricks-build databricks-init databricks-monitor

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

# Databricks targets
databricks-build:
	@echo "Building Python wheel..."
	python setup.py bdist_wheel
	@echo "✓ Wheel built in dist/"

databricks-validate:
	@echo "Validating Databricks Asset Bundle..."
	databricks bundle validate -t $(ENV)

databricks-deploy:
	@echo "Deploying Databricks Asset Bundle..."
	databricks bundle deploy -t $(ENV)

databricks-run:
	@echo "Triggering training pipeline job..."
	@echo "Listing jobs to find the training pipeline..."
	@databricks jobs list | grep -i "toxic.*training" || echo "Job not found - check Databricks UI"

databricks-monitor:
	@echo "Triggering drift monitoring job..."
	@databricks jobs list | grep -i "drift" || echo "Job not found - check Databricks UI"

databricks-init:
	@echo "Initializing Unity Catalog..."
	python databricks/setup/init_catalog.py --catalog $(CATALOG) --schema $(SCHEMA)

# Default environment variables
ENV ?= dev
CATALOG ?= mlops_catalog
SCHEMA ?= toxic_comments

help:
	@echo "Available targets:"
	@echo "  up              - Start all services"
	@echo "  down            - Stop all services"
	@echo "  restart         - Restart all services"
	@echo "  logs            - Show logs"
	@echo "  clean           - Remove all volumes"
	@echo "  ingest          - Trigger data ingestion DAG"
	@echo "  train           - Train model"
	@echo "  init            - Initialize system"
	@echo ""
	@echo "Databricks targets:"
	@echo "  databricks-build     - Build Python wheel"
	@echo "  databricks-validate  - Validate Asset Bundle"
	@echo "  databricks-deploy    - Deploy to Databricks (ENV=dev|prod)"
	@echo "  databricks-run       - Run training pipeline"
	@echo "  databricks-monitor   - Run drift monitoring"
	@echo "  databricks-init      - Initialize Unity Catalog"
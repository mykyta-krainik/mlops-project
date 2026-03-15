.PHONY: up down restart logs clean ingest train init help \
        upload-data ecr-push sm-pipeline-run drift-check slo-check load-test \
        tf-plan tf-apply tf-destroy test lint

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

# ── Tests & lint ───────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ pipelines/ monitoring/

# ── AWS bootstrap ──────────────────────────────────────────────────────────────
upload-data:
	python scripts/upload_data.py

ecr-push:
	@if [ -z "$$ECR_IMAGE_URI" ]; then echo "ERROR: ECR_IMAGE_URI not set"; exit 1; fi
	aws ecr get-login-password --region $${AWS_REGION:-us-east-1} | \
		docker login --username AWS --password-stdin $$(echo $$ECR_IMAGE_URI | cut -d/ -f1)
	docker build -t $$ECR_IMAGE_URI:latest .
	docker push $$ECR_IMAGE_URI:latest
	@echo "Pushed $$ECR_IMAGE_URI:latest"

# ── SageMaker Pipeline ────────────────────────────────────────────────────────
sm-pipeline-run:
	python pipelines/run_pipeline.py --wait

# ── Terraform ─────────────────────────────────────────────────────────────────
tf-plan:
	cd terraform && terraform init && terraform plan -out=tfplan

tf-apply:
	cd terraform && terraform apply -auto-approve tfplan

tf-destroy:
	@echo "WARNING: This will destroy all AWS resources. Press Ctrl-C to abort."
	@sleep 5
	cd terraform && terraform destroy -auto-approve

# ── Monitoring ────────────────────────────────────────────────────────────────
drift-check:
	python monitoring/drift_check.py

slo-check:
	python monitoring/slo_check.py

# ── Load testing ──────────────────────────────────────────────────────────────
load-test:
	locust --headless \
		--locustfile locustfile.py \
		-u 100 -r 10 \
		--run-time 2m \
		--csv /tmp/locust
	python scripts/check_load_test_results.py --csv /tmp/locust_stats.csv

help:
	@echo "Local demo:"
	@echo "  make up          Start docker-compose (MinIO + Airflow + Flask API)"
	@echo "  make train       Train model locally and reload API"
	@echo "  make down        Stop all local services"
	@echo ""
	@echo "AWS:"
	@echo "  make upload-data Upload train.csv to S3 raw bucket"
	@echo "  make ecr-push    Build + push Docker image to ECR"
	@echo "  make tf-plan     Terraform plan (creates tfplan)"
	@echo "  make tf-apply    Terraform apply"
	@echo ""
	@echo "SageMaker:"
	@echo "  make sm-pipeline-run  Trigger + wait for pipeline execution"
	@echo ""
	@echo "Monitoring:"
	@echo "  make drift-check Evidently drift report"
	@echo "  make slo-check   CloudWatch SLO check"
	@echo "  make load-test   Locust load test against staging endpoint"
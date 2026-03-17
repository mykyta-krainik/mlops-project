# Databricks Migration - Quick Start

This project has been migrated from Airflow to Databricks Lakeflow orchestration.

## 🚀 Quick Start

### 1. Configure Databricks

```bash
# Set up Databricks CLI
databricks configure --token

# Copy environment template
cp .env.example .env
# Edit .env with your Databricks credentials
```

### 2. Deploy to Databricks

```bash
# Build Python wheel
make databricks-build

# Validate configuration
make databricks-validate

# Deploy to dev environment
make databricks-deploy ENV=dev

# Initialize Unity Catalog (first time only)
make databricks-init
```

### 3. Run Pipeline

```bash
# Manual run
make databricks-run ENV=dev

# Or wait for scheduled run: Every Saturday at 2 AM UTC
```

## 📁 Project Structure

```
databricks/
├── tasks/              # Lakeflow pipeline tasks
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── challenger.py
│   └── deploy_model.py
├── monitoring/         # Drift detection
│   └── drift_detection.py
└── setup/             # Initialization scripts
    └── init_catalog.py

src/api/
├── app.py             # Flask API (updated with Databricks serving)
└── databricks_serving.py  # Databricks endpoint client
```

## 🔧 Configuration

### Model Type

Choose baseline or improved model via `MODEL_TYPE` in `.env`:

```bash
MODEL_TYPE=baseline  # Default hyperparameters
# or
MODEL_TYPE=improved  # Optimized hyperparameters + more features
```

### Serving

Enable Databricks serving in `.env`:

```bash
USE_DATABRICKS_SERVING=true
DATABRICKS_SERVING_ENDPOINT=https://your-workspace.cloud.databricks.com/serving-endpoints/toxic-comments-serving/invocations
```

## 📊 Pipeline Overview

1. **Data Ingestion**: Pull from Unity Catalog/S3 → Delta table
2. **Feature Engineering**: Text preprocessing + feature extraction
3. **Model Training**: Train baseline or improved model (configurable)
4. **Challenger**: Compare with production, promote if better (≥2% improvement)
5. **Deploy**: Update Databricks Model Serving endpoint

**Schedule**: Weekly on Saturday at 2 AM UTC

## 🔍 Monitoring

Drift detection runs weekly (Sunday 3 AM) and monitors:
- Text embedding drift
- Text length distribution
- Toxicity distribution
- Class balance
- Prediction confidence

View reports:
```sql
SELECT * FROM mlops_catalog.toxic_comments.drift_reports
ORDER BY timestamp DESC;
```

## 🧪 Testing

### Test Local API

```bash
# Start API
python -m src.api.app

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "This is a test"}'

# Check serving status
curl http://localhost:5000/serving/status
```

### Test Databricks Endpoint

```bash
curl -X POST $DATABRICKS_SERVING_ENDPOINT \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["test comment"]}'
```

## 📚 Documentation

- **[Setup Guide](docs/databricks_setup.md)**: Detailed setup instructions
- **[Walkthrough](walkthrough.md)**: Complete implementation walkthrough
- **[Implementation Plan](implementation_plan.md)**: Technical design decisions

## 🎯 Key Features

- ✅ **Configurable Training**: Single MODEL_TYPE flag for baseline/improved
- ✅ **Smart Promotion**: Automatic challenger-based model promotion
- ✅ **Robust Serving**: Databricks endpoint with local fallback
- ✅ **Drift Detection**: Evidently-based monitoring
- ✅ **Weekly Automation**: Scheduled pipeline runs
- ✅ **Asset Bundle**: Reproducible Databricks deployments

## 🔄 Migration from Airflow

The original Airflow DAG has been replaced with Databricks Lakeflow. The Airflow setup remains in the repository for reference but is no longer actively used.

**Key Changes**:
- Airflow DAG → Databricks Lakeflow workflow
- Minio → Unity Catalog / Delta tables
- Local training → Databricks job clusters
- Local serving → Databricks Model Serving (with local fallback)
- No monitoring → Evidently drift detection

## 🛠️ Makefile Commands

```bash
make databricks-build      # Build Python wheel
make databricks-validate   # Validate Asset Bundle
make databricks-deploy     # Deploy to Databricks
make databricks-run        # Run training pipeline
make databricks-monitor    # Run drift monitoring
make databricks-init       # Initialize Unity Catalog
```

## 📞 Support

For issues or questions, refer to:
1. [Databricks Setup Guide](docs/databricks_setup.md)
2. [Walkthrough Document](walkthrough.md)
3. Databricks workspace job logs

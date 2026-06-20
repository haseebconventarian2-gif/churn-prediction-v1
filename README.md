<div align="center">

# Telco Churn Prediction v1

Telco customer churn prediction system with exploratory analysis, modular ML pipelines, and a FastAPI web application.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Reference%20Implementation-6366F1)

</div>

---

## Overview

Telco customer churn prediction system with exploratory analysis, modular ML pipelines, and a FastAPI web application.

## 📖 The Story

This repository captures an earlier chapter of the Telco churn project: the move from exploratory analysis toward a modular prediction system. The goal was to turn notebook discoveries into code that could be rerun, tested, and eventually served to an application.

The project separates ingestion, validation, transformation, training, evaluation, and prediction concerns. Configuration files describe the expected data, saved artifacts connect training to inference, and FastAPI provides a simple way to test the resulting model through a browser.

As a version-one implementation, it is valuable because the architecture is visible and approachable. The next iteration can remove duplicated folders, add tests and experiment tracking, and compare the pipeline against the newer churn repository.

## Highlights

- Exploratory data analysis
- Reusable preprocessing and training pipeline
- Model evaluation and saved artifacts
- FastAPI prediction interface

## Tech Stack

Python Â· pandas Â· scikit-learn Â· imbalanced-learn Â· FastAPI

## Getting Started

```bash
git clone https://github.com/haseebconventarian2-gif/churn-prediction-v1.git
cd churn-prediction-v1
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

No external API credentials are required.

> Store credentials in `.env` and never commit secrets.

## Run

```bash
python main.py

# Start the prediction dashboard
uvicorn web_app_fastapi:app --reload
```

## Project Status

This is a learning and reference implementation. Review security, validation, monitoring, and deployment settings before production use.

## Detailed Code Reference

**Runtime flow:** `Dataset -> validation -> transformation -> training -> evaluation -> artifacts -> API`

### Repository map

- `__pycache__/` - supporting package or resources
- `api/` - supporting package or resources
- `components/` - supporting package or resources
- `config/` - supporting package or resources
- `config - Copy/` - supporting package or resources
- `configuration/` - supporting package or resources
- `constants/` - supporting package or resources
- `entity/` - supporting package or resources
- `experiments/` - supporting package or resources
- `main.py` - project file
- `pipeline/` - supporting package or resources
- `README.md` - project file
- `requirements.txt` - project file
- `scripts/` - supporting package or resources
- `Telco_Churn_EDA.ipynb` - project file
- `templates/` - supporting package or resources
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - project file
- `web_app_fastapi.py` - project file

### Validation checklist

1. Install dependencies in a clean virtual environment.
2. Configure only the environment variables needed by enabled integrations.
3. Start the documented entry point and test its health or root route.
4. Exercise successful and invalid requests.
5. Confirm secrets, private datasets, indexes, and model artifacts are ignored.

### Production checklist

- Use managed secret storage.
- Add authentication, authorization, rate limiting, and request-size limits.
- Add automated tests, structured logs, monitoring, and health checks.
- Pin and audit dependencies.
- Define retention and privacy controls for audio and customer data.

> This README reflects the current codebase. External AI, telephony, and messaging features require their respective accounts, assets, and approvals.



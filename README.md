<div align="center">

# Telco Churn Prediction v1

Telco customer churn prediction system with exploratory analysis, modular ML pipelines, and a FastAPI web application.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Reference%20Implementation-6366F1)

</div>

---

## Overview

Telco customer churn prediction system with exploratory analysis, modular ML pipelines, and a FastAPI web application.

## Highlights

- Exploratory data analysis
- Reusable preprocessing and training pipeline
- Model evaluation and saved artifacts
- FastAPI prediction interface

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · FastAPI

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

<!-- code-audit-details -->

## 🔄 Runtime Flow

`Dataset → validation → transformation → training → evaluation → artifacts → prediction API`

This flow is derived from the current entry points and service calls.

## 🗂 Code Map

| Path | Responsibility |
| --- | --- |
| `__pycache__/` | Supporting resource |
| `api/` | Supporting resource |
| `components/` | Supporting resource |
| `config/` | Supporting resource |
| `config - Copy/` | Supporting resource |
| `configuration/` | Supporting resource |
| `constants/` | Supporting resource |
| `entity/` | Supporting resource |
| `experiments/` | Supporting resource |
| `main.py` | Application entry point |
| `pipeline/` | Supporting resource |
| `requirements.txt` | Python dependencies |
| `scripts/` | Supporting resource |
| `templates/` | Supporting resource |
| `web_app_fastapi.py` | Prediction web application |

## 🔐 Environment Variables

No environment-variable reads were detected.

## 🌐 Detected API Routes

| Method | Endpoint |
| --- | --- |
| `GET` | `/` |
| `GET` | `/health` |
| `GET` | `/media/{media_id}` |
| `GET` | `/webhook` |
| `GET` | `/whatsapp/diagnose` |
| `POST` | `/audio` |
| `POST` | `/predict` |
| `POST` | `/text` |
| `POST` | `/webhook` |
| `POST` | `/whatsapp/push` |

## 🧪 Validation Guide

1. Install dependencies in a clean virtual environment.
2. Start the documented entry point and test the root or health route.
3. Exercise one valid and one invalid request.
4. Verify external-service errors are handled clearly.
5. Confirm secrets, private data, indexes, and model artifacts are ignored.

## 🔒 Production Checklist

- Use managed secret storage and rotate exposed credentials.
- Add authentication, authorization, rate limiting, and request-size limits.
- Add automated tests, structured logging, monitoring, and health checks.
- Pin and audit dependencies.
- Define retention and privacy controls for audio and customer data.

## ⚠️ Code-Audit Notes

- Documentation reflects the current repository code and may expose integrations that need separate cloud accounts, model assets, or channel approval.
- Treat the project as a reference implementation until its security and deployment configuration are hardened.

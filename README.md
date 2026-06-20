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

# Project: Building Energy Consumption Prediction
## Seattle Carbon Neutrality Strategy - Predictive Modeling

[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow.svg)](https://www.python.org/)

###  Project Context
The city of Seattle aims to become carbon-neutral by 2050. This project uses 2016 benchmarking data to develop predictive models for energy consumption and CO2 emissions of non-residential buildings, helping the city target its sustainability efforts.

###  Business & Technical Objectives
- **Predict Energy Needs**: Estimate total energy consumption for non-residential buildings.
- **Estimate Emissions**: Predict greenhouse gas emissions (CO2) based on building characteristics.
- **Feature Evaluation**: Assess the impact of the **ENERGY STAR Score** on model accuracy to determine if its collection cost is justified.

###  Technical Architecture
1. **EDA & Preprocessing**: In-depth analysis of building types, usage, and location.
2. **Feature Engineering**: Creation of relevant variables for energy prediction.
3. **Hyperparameter Tuning**: Optimization using GridSearchCV and RepeatedKFold.
4. **Evaluation**: Performance assessment using R², MAE, and RMSE.

---

###  Quick Start (Docker)

#### 1. Prerequisites
- Docker Desktop
- Docker Compose V2

#### 2. Launch the System
```bash
docker-compose up --build
```

#### 3. Access the Services
- **Jupyter Notebook**: [http://localhost:8884](http://localhost:8884) (Open mission4.ipynb)

---

###  Project Structure
```text
 mission4.ipynb       # Main analysis and modeling notebook
 src/
    scripts/         # Data processing, analysis, and visualization
    utils/           # Caching and utility functions
 dataset/             # Seattle Building Energy Benchmarking data
 docker-compose.yml   # Container orchestration
 Dockerfile           # Python environment
```

###  Key Insights
- **Size and Usage Matter**: Building floor area and primary usage type are the most significant predictors of energy consumption.
- **Non-Linearity**: Energy consumption patterns are highly non-linear, making log-transformations and tree-based models (Gradient Boosting, Random Forest) highly effective.
- **ENERGY STAR Value**: While the ENERGY STAR Score improves model performance, robust predictions are still possible using only physical building attributes.

---
*This project demonstrates the application of machine learning to urban sustainability and energy efficiency challenges.*

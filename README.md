# US Overdose Death Trends (2015-2025): Forecasting and Early Warning System

## Overview
This project analyzes US drug overdose death trends from 2015-2025, with a focus on:
- Synthetic opioids (fentanyl)
- Psychostimulants (methamphetamine)
- Cocaine
- State-level shifts over time

The long-term end goal is to build a state-level overdose forecasting model and early warning system.

## Motivation and Problem Context
Drug overdose deaths remain a major public health crisis in the United States. In recent years, trends have shifted dramatically due to:
- Rapid spread of fentanyl
- Rising stimulant-related deaths
- Drug combinations increasing lethality

Understanding these patterns and forecasting near-future risk can support:
- Resource allocation
- Public health interventions
- Prevention and harm-reduction planning

## Data Source
This project uses the CDC dataset:

VSRR Provisional Drug Overdose Death Counts (Monthly, 12-month rolling totals)

The dataset includes:
- 50,000+ rows
- State-level and national-level breakdowns
- Multiple drug categories
- Completeness and reporting-delay metadata

## End Goal (Final Phase)
State-level overdose forecasting and early warning system:
1. Forecast overdose deaths by state (1-3 months ahead)
2. Predict category-specific outcomes:
- Synthetic opioids (fentanyl)
- Psychostimulants
- Cocaine
- Total overdose deaths
3. Produce an interpretable risk ranking of states
4. Explain drivers of predicted risk (feature importance and explainability)

## ML Training Pipeline (Current)
The project now includes a training pipeline in `model_training/` that runs three regression algorithms on the cleaned dataset:
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Split Strategy
- Chronological split by date (no shuffling)
- Train: 70%
- Validation: 15%
- Test: 15%

### Run Training
```bash
python model_training/train_models.py
```

### Results
Outputs are saved in `results/model_training/`:
- `split_metadata.json`
- `metrics_summary.json`
- `best_model.txt`
- `training_summary.md`
- `predictions_by_model.json`

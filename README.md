# F1 Race Prediction: Predicting Podium Finishers

A machine learning project that uses 2020-2024 Formula 1 historical data to predict which drivers will finish on the podium.

## Problem Definition

Predicting podium finishes (top 3 positions) in Formula 1 races is a binary classification problem with real-world sports analytics applications. This project explores which factors -- driver skill, constructor quality, grid position, circuit characteristics, and weather conditions -- are most influential in determining podium outcomes.

**Target Variable**: `podium_finish` (1 if the driver finishes P1-P3, 0 otherwise)

## Dataset

- **Source**: Simulated F1 race data modeled after real 2020-2024 season patterns
- **Scope**: 103 Grands Prix, ~2,140 individual race entries (20+ drivers per race)
- **Features**: 24 columns including driver stats, constructor performance, grid position, circuit characteristics, weather, and tire compounds

### Key Data Characteristics
- Reflects Red Bull/Verstappen dominance in 2021-2024
- Captures Mercedes' 2020 supremacy and McLaren's 2023-2024 resurgence
- Street circuits (Monaco, Singapore, Baku) modeled with reduced overtaking
- ~15% DNF (Did Not Finish) rate
- Class imbalance: ~15% podium vs ~85% non-podium

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Programming language |
| pandas / numpy | Data manipulation |
| scikit-learn | Model building, evaluation, preprocessing |
| XGBoost | Gradient boosting classifier |
| matplotlib / seaborn | Data visualization |
| Jupyter Notebook | Analysis and documentation |

## Project Structure

```
f1-race-prediction/
├── data/
│   └── f1_data.csv              # Simulated race dataset (2020-2024)
├── notebooks/
│   └── f1_prediction.ipynb      # Main analysis notebook
├── models/                       # Directory for saved models
├── outputs/
│   └── figures/                  # Generated visualizations
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/f1-race-prediction.git
   cd f1-race-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook notebooks/f1_prediction.ipynb
   ```

4. **Run all cells** to reproduce the full analysis pipeline.

## Notebook Sections

The `f1_prediction.ipynb` notebook covers the complete ML workflow:

1. **Problem Definition & Setup** -- Load data, define the prediction task
2. **Data Cleaning & Feature Engineering** -- Handle missing values, create derived features
3. **Exploratory Data Analysis** -- Visualize distributions, relationships, and patterns
4. **Feature Selection & Preprocessing** -- Train/test split (2020-2023 / 2024), scaling, class weights
5. **Model Building & Comparison** -- Logistic Regression, Random Forest, XGBoost with cross-validation
6. **Best Model Evaluation** -- Confusion matrix, ROC curves, feature importance
7. **Key Insights** -- Constructor advantage, circuit type effects, top performers by era
8. **Conclusions** -- Key findings, limitations, and future improvements

## Model Performance Summary

Three models were trained and evaluated on the 2024 season (held-out test set):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.85 | ~0.45 | ~0.70 | ~0.55 | ~0.90 |
| Random Forest | ~0.86 | ~0.48 | ~0.65 | ~0.55 | ~0.90 |
| XGBoost | ~0.87 | ~0.50 | ~0.60 | ~0.55 | ~0.91 |

> Note: Exact metrics may vary slightly due to the random nature of the simulated data.

## Key Findings

1. **Grid position is the single strongest predictor** of podium finishes. Drivers starting from the front row (P1-P2) have a ~60-70% chance of finishing on the podium, while those outside the top 6 have less than a 20% chance.

2. **Constructor quality dominates individual driver skill.** Top-tier teams (Red Bull, Mercedes, Ferrari, McLaren) account for over 95% of all podium finishes across the dataset.

3. **Circuit type affects race dynamics.** Street circuits like Monaco, Singapore, and Baku show less position change from grid to finish, making qualifying performance even more critical.

4. **Era-specific dominance patterns are clearly visible.** Mercedes led in 2020, Red Bull/Verstappen controlled 2021-2023, and McLaren emerged as a serious contender in 2024.

## Generated Visualizations

All plots are saved to `outputs/figures/`:

- Target variable distribution
- Podiums by constructor and driver
- Grid position vs podium analysis
- Circuit type and weather effects
- Model comparison charts
- ROC curves
- Feature importance plot
- Constructor advantage across seasons

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with Python, scikit-learn, and XGBoost. Data is simulated for educational purposes.*

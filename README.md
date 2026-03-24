# Titanic Kaggle Competition

A production-quality solution to the [Kaggle Titanic](https://www.kaggle.com/c/titanic) "Machine Learning from Disaster" competition. The project trains multiple classifiers, selects the best via 5-fold cross-validation, and generates a Kaggle-ready submission CSV.

---

## Project Structure

```
titanic-kaggle/
├── data/
│   ├── raw/           # Kaggle source files (train.csv, test.csv)
│   └── processed/     # Intermediate processed files
├── models/            # Serialised best model (best_model.pkl)
├── notebooks/
│   └── 01_eda.ipynb   # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── features.py    # Feature engineering pipeline
│   ├── train.py       # Model training + selection
│   └── predict.py     # Submission generation
├── submissions/       # Generated submission CSVs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/titanic-kaggle.git
cd titanic-kaggle
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

```bash
kaggle competitions download -c titanic -p data/raw/
unzip data/raw/titanic.zip -d data/raw/
```

> Requires a valid `~/.kaggle/kaggle.json` API token. See the
> [Kaggle API docs](https://github.com/Kaggle/kaggle-api) for setup instructions.

---

## Usage

### Exploratory Data Analysis

Launch Jupyter and open the EDA notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The notebook covers:
- Data shape, dtypes, `.describe()`
- Missing values heatmap
- Overall survival rate
- Survival by Sex, Pclass, and Embarked
- Age & Fare distributions
- Correlation heatmap
- Family size analysis
- Title extraction analysis

### Train models

```bash
python3 src/train.py
```

Trains five classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM) using 5-fold stratified cross-validation, prints CV accuracy for each, selects the best, retrains on the full training set, and saves the model to `models/best_model.pkl`.

### Generate submission

```bash
python3 src/predict.py
```

Loads the saved model and test data, generates predictions, and writes a timestamped CSV to `submissions/submission_YYYYMMDD_HHMMSS.csv`.

---

## Feature Engineering

All feature engineering lives in `src/features.py`:

| Feature | Description |
|---|---|
| `Title` | Extracted from name, normalised to Mr / Mrs / Miss / Master / Rare |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if FamilySize == 1 |
| `AgeGroup` | Binned: Child / Teen / YoungAdult / Adult / Senior |
| `FareBin` | Quartile bins: Low / Medium / High / VeryHigh |
| Missing `Age` | Imputed with median grouped by Pclass + Sex |
| Missing `Embarked` | Imputed with mode |
| Missing `Fare` | Imputed with median |

Dropped columns: `Name`, `Ticket`, `Cabin`, `PassengerId` (retained in test for submission assembly).

---

## Models

| Model | Library |
|---|---|
| LogisticRegression | scikit-learn |
| RandomForestClassifier | scikit-learn |
| GradientBoostingClassifier | scikit-learn |
| XGBClassifier | xgboost |
| LGBMClassifier | lightgbm |

Selection criterion: mean accuracy over 5-fold stratified cross-validation.

---

## Results

| Model | CV Accuracy (mean ± std) |
|---|---|
| LogisticRegression | — |
| RandomForest | — |
| GradientBoosting | — |
| XGBoost | — |
| LightGBM | — |

> Run `python3 src/train.py` to populate the results above.

---

## License

MIT

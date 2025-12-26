# Sovereign Risk Prediction using Machine Learning

**Project**: Systematic comparison of 29 ML models to predict sovereign credit ratings  
**Dataset**: 950 observations, 38 countries, 8 economic indicators, 20 rating classes  
**Result**: Random Forest (79.68% accuracy) is the best model

---

### **Option 1: Installation with Conda (Recommended)**

#### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/DSAP-Project.git
cd DSAP-Project
```

#### 2. Create Conda Environment
```bash
# Create environment from environment.yml file
conda env create -f environment.yml
```

#### 3. Activate Environment
```bash
conda activate dsap-project
```

#### 4. Verify Installation
```bash
# Verify that all dependencies are installed
python -c "import pandas, numpy, sklearn, tensorflow, keras; print('✓ All packages installed successfully')"
```

#### 5. Run the Project
```bash
# Option 1: Complete pipeline (38 steps, ~15-20 min)
python main.py

# Option 2: Interactive notebooks
jupyter notebook
```

#### 6. Deactivate Environment (when finished)
```bash
conda deactivate
```

---

### **Option 2: Installation with pip**

#### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/DSAP-Project.git
cd DSAP-Project
```

#### 2. Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Project
```bash
# Option 1: Complete pipeline
python main.py

# Option 2: Interactive notebooks
jupyter notebook
```

---

## 📁 Project Structure

```
DSAP-Project/
├── main.py                    # Automated pipeline (38 steps)
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/                   # Raw data (CSV)
│   └── processed/             # Cleaned data
│
├── src/                       # Source code
│   ├── data_loader.py         # Data preprocessing
│   ├── models.py              # Phase 1: Regression
│   ├── classification.py      # Phase 2: Classification
│   ├── unsupervised.py        # Phase 3: K-Means + PCA
│   └── deep_learning.py       # Phase 4: MLP
│
├── notebooks/                 # 5 interactive notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_regression.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_unsupervised.ipynb
│   └── 05_deep_learning.ipynb
│
└── results/                   # Results (CSV, PNG)
    ├── regression_metrics.csv
    ├── classification_metrics.csv
    ├── clustering/
    ├── pca/
    └── deep_learning/
```

---

## 🎯 Methodology (4 Phases)

### Phase 1: Regression (13 models)
- Linear, Polynomial, Ridge Regression
- **Best**: Ridge + Polynomial (R²=0.6477)

### Phase 2: Classification (11 models)
- k-NN, Naive Bayes, Decision Tree, Random Forest
- **Best**: Random Forest n=200 (79.68% accuracy) 🏆

### Phase 3: Unsupervised Learning
- K-Means: 3 economic clusters (ARI=0.07 with ratings)
- PCA: 5 components explain 80% variance

### Phase 4: Deep Learning (6 models)
- MLP Simple (55.79%) vs MLP Improved (70.53%)
- Optimizers, Learning Rates, L2 Regularization tested
- **Conclusion**: Random Forest > Deep Learning for this dataset

---

## 📊 Main Results

### Top 5 Models
1. **Random Forest (n=200)**: 79.68% ✅
2. Random Forest (n=100): 78.95%
3. Random Forest (n=50): 78.42%
4. MLP Improved: 70.53%
5. Decision Tree (depth=10): 69.47%

### Top 3 Features (Importance)
1. **FX_Reserves**: 16.97%
2. **Public_Debt**: 15.45%
3. **Unemployment**: 13.86%

### Key Insights
- ✅ Classification > Regression (+15%)
- ✅ Random Forest > Deep Learning (+9%)
- ✅ Ensemble methods > Individual models (+10%)
- ⚠️ Deep Learning requires >5,000 observations (dataset too small)

---

## 📚 Complete Documentation

- **README.md**: This file (quick guide)
- **PROPOSAL.md**: Detailed project proposal

---

## 🎓 Technologies Used

- Python 3.12
- Scikit-learn 1.3.0 (Classical ML)
- TensorFlow 2.20.0 (Deep Learning)
- Pandas, NumPy (Data processing)
- Matplotlib, Seaborn (Visualizations)

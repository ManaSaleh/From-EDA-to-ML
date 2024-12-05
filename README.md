# **From EDA to ML**

## **Project Overview**
This project demonstrates the complete process of transitioning from Exploratory Data Analysis (EDA) to Machine Learning (ML) using decision tree-based models. It includes data cleaning, feature engineering, model training, evaluation, and boosting techniques. The focus is on creating robust and interpretable models using Python.

---

## **Project Structure**

```plaintext
tree
.
├── Data
│   └── train.csv                # Raw dataset for training
├── Model
│   ├── dt_model.pkl             # Saved Decision Tree model
│   └── voting_model.pkl         # Saved Voting Classifier model
└── Notebook
    ├── EDA-For-ML.ipynb         # Notebook covering EDA and ML steps
    └── expline.ipynb            # Detailed explanations of methods and steps
```

---

## **1. Data Cleaning**
- **Goal**: Ensure the dataset is clean and ready for analysis.
- **Steps**:
  - Handle missing values:
    - `Age`: Imputed using median values.
    - `Cabin`: Encoded to indicate whether a cabin is known.
    - `Embarked`: Imputed using the mode.
  - Verified data types and corrected them where necessary.
  - Removed duplicates and unnecessary characters.

---

## **2. Exploratory Data Analysis (EDA)**
- **Objective**: Understand the dataset and relationships between variables.
- **Key Analyses**:
  - Visualized distributions of numerical columns (`Age`, `Fare`).
  - Analyzed categorical features (`Pclass`, `Sex`, `Embarked`) using bar plots.
  - Examined correlations between features and the target variable (`Survived`).
  - Generated insights to inform feature engineering.

---

## **3. Feature Engineering**
- Created new features like `FamilySize` by combining `SibSp` and `Parch`.
- Applied transformations:
  - Log transformation for skewed data (`Fare`).
  - Square root transformation for moderate skew (`Age`).

---

## **4. Multicollinearity Handling**
- Identified highly correlated features using a correlation matrix.
- Removed redundant features to improve model interpretability.

---

## **5. Handling Outliers**
- Used Z-scores and the IQR method to detect outliers.
- Applied capping to limit the impact of extreme values.

---

## **6. Scaling Data**
- Standardized numerical features to improve model performance:
  - Ensured features had a mean of 0 and standard deviation of 1.

---

## **7. Imbalanced Data**
- Checked class imbalance in the `Survived` target variable.
- Applied balancing techniques:
  - **Oversampling** using SMOTE.
  - **Undersampling** to reduce bias from the majority class.

---

## **8. Machine Learning Models**
### **Simple Decision Tree**
- Trained using the `DecisionTreeClassifier` with hyperparameter tuning.
- Evaluated using:
  - **Accuracy**: Overall performance of the model.
  - **Classification Report**: Precision, recall, and F1-score for both classes.

### **Voting Classifier**
- Combined multiple Decision Tree classifiers using majority voting.
- Steps:
  1. Created balanced subsets for diversity.
  2. Trained individual classifiers (`tree1`, `tree2`, `tree3`).
  3. Combined predictions using a hard voting strategy.

### **Boosting**
- Compared three boosting models:
  - **Gradient Boosting**: Sequential model with a focus on reducing loss.
  - **XGBoost**: Optimized version of gradient boosting with regularization.
  - **LightGBM**: Lightweight and efficient boosting technique.
- Each model was evaluated for accuracy and robustness.

---

## **10. How to Run the Code**
### Prerequisites
- Python
- Required Libraries:
  - `pandas`, `numpy`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/project-name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks:
   - Open `EDA-For-ML.ipynb` to review the EDA and ML process.
   - Use `expline.ipynb` for detailed explanations of methods and steps.
4. Train and save models:
   ```bash
   python train_models.py
   ```
5. Evaluate results:
   ```bash
   python evaluate.py
   ```

---

## **11. References**
1. Scikit-learn Documentation: https://scikit-learn.org/
2. XGBoost Guide: https://xgboost.readthedocs.io/
3. LightGBM Documentation: https://lightgbm.readthedocs.io/
4. Gradient Boosting: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting

---
---

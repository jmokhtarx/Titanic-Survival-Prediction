#  Titanic - Machine Learning from Disaster

This project is a solution to the famous [Kaggle Titanic competition](https://www.kaggle.com/c/titanic), where the goal is to predict whether a passenger survived or not using machine learning techniques.

---

##  Project Overview

- **Objective**: Predict survival on the Titanic using passenger data (e.g., age, gender, class).
- **Approach**: Data preprocessing using custom transformers, feature engineering, model training with `RandomForestClassifier`, and hyperparameter tuning using `GridSearchCV`.
- **Tools**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

##  Technologies & Libraries

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

##  Project Workflow

### 1. **Data Loading & Exploration**
- Load `train.csv` and `test.csv`
- Explore data using `.info()`, `.describe()`, and visualizations like correlation heatmaps

### 2. **Stratified Sampling**
- Use `StratifiedShuffleSplit` to split training data while maintaining class balance on `Survived`, `Pclass`, and `Sex`.

### 3. **Preprocessing Pipeline**
Custom transformers created for:
- **Imputing Age** (with mean)
- **Encoding** categorical features (`Sex`, `Embarked`) using `OneHotEncoder`
- **Dropping** irrelevant features (e.g., `Name`, `Ticket`, `Cabin`, etc.)

### 4. **Model Building**
- Model: `RandomForestClassifier`
- Hyperparameter tuning using `GridSearchCV`
- Training done on both train-validation and full datasets

### 5. **Prediction & Submission**
- Preprocess `test.csv` data using the pipeline
- Predict survival
- Save results as `predictions.csv`

---

##  Model Performance

- **Validation Accuracy**: Evaluated using `grid_search.score()` on the test split
- **Final Model**: Tuned with best parameters from GridSearch on full dataset

---

##  Project Structure
-train.csv
-test.csv
-Titanic_Model.ipynb
-predictoins.csv
-README.md
---

##  Results Sample

PassengerId	survived
	892 |	0
	893	| 0
	894	| 0
	895	| 0
	896	| 1
	897	| 0
	898	| 1
	899	| 0
	900	| 1
	901	| 0


---

## 🚀 Future Improvements

- Use more advanced models (e.g., XGBoost, LightGBM)
- Feature importance analysis
- Cross-validation visualization
- Handle missing values in a more domain-informed way

---

## 👤 Author

- *Ahmed Mokhtar*


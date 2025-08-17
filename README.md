# Titanic Survival Prediction

## Problem Statement and Goal

The goal of this project is to predict whether a passenger survived the Titanic disaster using demographic and travel-related features. This binary classification problem, sourced from the Kaggle Titanic competition, demonstrates my ability to implement a complete machine learning pipeline, including data preprocessing, feature engineering, model training, and evaluation, tailored to showcase my skills for a professional portfolio.

## Solution Approach

The project follows a structured machine learning workflow:
- **Exploratory Data Analysis (EDA)**: Visualized feature distributions and correlations using heatmaps and pair plots to identify patterns.
- **Data Preprocessing**: Handled missing values with `KNNImputer`, encoded categorical variables, and standardized features using `StandardScaler`.
- **Feature Engineering**: Created features like `FamilySize`, `IsAlone`, `CategoricalFare`, and `Title` to enhance predictive power.
- **Model Training**: Evaluated multiple classification algorithms:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- **Model Evaluation**: Used accuracy, F1-score, classification reports, and confusion matrices for performance assessment.
- **Hyperparameter Tuning**: Applied `GridSearchCV` to optimize the KNN model.
- **Cross-Validation**: Employed K-Fold cross-validation for robust model evaluation.

The K-Nearest Neighbors (KNN) model was selected as the best performer based on validation accuracy.

## Technologies & Libraries

- **Python 3.9+**: Core programming language
- **pandas**, **numpy**: Data manipulation and analysis
- **matplotlib**, **seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms, preprocessing, and evaluation
- **xgboost**: Gradient boosting for enhanced predictions

## Description about Dataset

The dataset, sourced from the Kaggle Titanic competition, consists of two files:
- **train.csv**: 891 passenger records with features:
  - `PassengerId`: Unique identifier for each passenger
  - `Survived`: Target variable (0 = Did not survive, 1 = Survived)
  - `Pclass`: Passenger class (1 = First, 2 = Second, 3 = Third)
  - `Name`: Passenger's name
  - `Sex`: Gender (male or female)
  - `Age`: Age of the passenger
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Ticket`: Ticket number
  - `Fare`: Ticket fare
  - `Cabin`: Cabin number (often missing)
  - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **test.csv**: 418 passenger records with the same features (excluding `Survived`) for predictions.

Missing values are prevalent in `Age` (19.9% missing) and `Cabin` (77.1% missing), requiring careful preprocessing.

## Installation & Execution Guide

### Prerequisites
- Python 3.9+
- Jupyter Notebook
- Datasets: `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data)

### Installation
Install required packages using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Execution
1. Download `train.csv` and `test.csv` from Kaggle.
2. Place the datasets in the same directory as the notebook.
3. Run the notebook:
   ```bash
   jupyter notebook "Project.ipynb"
   ```
4. The notebook generates a `submission.csv` file with test set predictions.

## Key Results / Performance

- **Best Model**: K-Nearest Neighbors (KNN) achieved the highest validation accuracy among tested models.
- **Evaluation Metrics**: Models were assessed using accuracy, F1-score, classification reports, and confusion matrices, with detailed results in the notebook.
- **Cross-Validation**: 5-fold cross-validation ensured robust performance estimates.
- **Output**: Predictions for the test set were saved in `submission.csv` for Kaggle submission.

## Screenshots / Some Sample Outputs

The notebook includes:
- **Correlation Heatmaps**: Visualizing feature relationships.
- **Pair Plots**: Showing feature distributions and survival patterns.
- **Confusion Matrices**: Displaying classification performance for each model.

> 💡 *Some interactive outputs (e.g., plots) may not display correctly on GitHub. Please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## Additional Learnings / Reflections

This project enhanced my skills in:
- Feature engineering, including extracting meaningful features like `Title` and `FamilySize`.
- Handling missing data with advanced techniques like `KNNImputer`.
- Comparing multiple machine learning models to select the best performer.
- Applying hyperparameter tuning and cross-validation for robust model evaluation.
- Using visualizations to communicate insights effectively.

The project underscored the importance of preprocessing, particularly for handling missing values and scaling features for distance-based algorithms like KNN.

## 👤 Author

**Mehran Asgari**  
**Email**: [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)  
**GitHub**: [https://github.com/imehranasgari](https://github.com/imehranasgari)

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.
# Bank Term Deposit Prediction (Classification Project)

## Project Overview
The goal of this project is to build a machine learning model to predict whether a banking client will subscribe to a term deposit. 

## Dataset
The project utilizes the **Bank Marketing (Additional Full)** dataset available on Kaggle:
[Bank Additional Full Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv)

---

## Exploratory Data Analysis (EDA)

### Categorical Variables
* **Missing values:** All categorical columns are complete and contain no missing values.
* **Cardinality:** Except for `job`, `education`, and `month`, most columns have a low number of unique values. This simplifies the choice of encoding methods.
* **Encoding strategies:** 
  * `month` and `day_of_week` can be encoded sequentially based on their calendar order.
  * `education` can be ordinally encoded from 1 to 8 based on the academic level (from *unknown* to *university degree*).
* **Feature impact:** Columns `housing`, `loan`, and `day_of_week` show no significant impact on the target variable.
* **Handling imbalanced categories:** The `default` column contains only 3 "yes" values; these rows can either be dropped or merged into the *unknown* category.
* **Key categorical features:** `poutcome`, `job`, `education`, `default`, `contact`, and `month` were identified as the most significant predictors.

### Numerical Variables
* `pdays`: Replaced the value `999` (which indicates no previous contact) with `0`, and all other values with `-1` to represent binary communication status.
* `campaign`: Handled extreme outliers by replacing them with the median value.
* `euribor3m`: Shows a wide data distribution, which was kept as-is for the initial iteration.
* `nr.employed`: Contains the largest absolute values among all numerical features and requires scaling/normalization.
* **Multicollinearity:** Strong correlations were detected between `euribor3m`, `emp.var.rate`, and `nr.employed`. For linear/regression models, only one of these features should be retained.

---

## Model Evaluation Metric
Due to a significant class imbalance in the target variable (the number of clients who opened a deposit is much smaller than those who did not), **F1-Score** was selected as the primary evaluation metric. The main focus is the prediction quality for the positive class.

---

## Data Preprocessing
To satisfy the requirements of different algorithms (especially Linear Regression and KNN), the following data transformations were performed:

1. Applied categorical transformations based on EDA insights (`month` and `day_of_week` sequential encoding, `education` ordinal encoding, `default` cleaning).
2. Scaled numerical features using `StandardScaler`.
3. Encoded remaining categorical features using `OneHotEncoder`.
4. Trained baseline models, followed by Tree-based models and Gradient Boosting.

---

## Model Performance & Results

After hyperparameter tuning and optimization, the models achieved the following results:

| Model Name | Train F1 | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Randomized Search LGBM** | **0.522** | **0.530** | **0.475** | **0.599** |
| Baseline LGBM | 0.563 | 0.530 | 0.476 | 0.597 |
| RandomForest | 0.525 | 0.525 | 0.457 | 0.617 |
| Hyperopt LGBM | 0.515 | 0.518 | 0.457 | 0.599 |
| Logistic Regression | 0.434 | 0.457 | 0.352 | 0.650 |
| KNN | 0.476 | 0.368 | 0.549 | 0.277 |

---

## Conclusion
The **Randomized Search LightGBM (LGBM)** model emerged as the top performer.

* **Highest Test F1 (0.5297):** Demonstrates the best overall balance between Precision and Recall with the fewest cumulative errors.
* **Precision (~0.475):** For a bank, this implies that almost every second marketing call based on the model's prediction will be successful.
* **Generalization:** The F1-score on the test set (0.530) is slightly higher than on the training set (0.522), indicating stable generalization without overfitting.

### Future Work
To further improve model performance, future iterations will focus on advanced **Feature Engineering** to create new, high-quality domain-specific features.

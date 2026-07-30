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
  * `education` can be ordinally encoded from 0 to 6 based on the academic level (from *unknown* to *university degree*).
* **Feature impact:** Columns `housing`, `loan`, and `day_of_week` show no significant impact on the target variable.
* **Handling imbalanced categories:** The `default` column contains only 3 "yes" values; these rows can either be dropped or merged into the *unknown* category.
* **Key categorical features:** `poutcome`, `job`, `education`, `default`, `contact`, and `month` were identified as the most significant predictors.

### Numerical Variables
* `pdays`: Replaced the value `999` (which indicates no previous contact) with `0`, and all other values with `- 1` to represent binary communication status.
* `campaign`: Handled extreme outliers by replacing them with the median value.
* `euribor3m`: Shows a wide data distribution, which was kept as-is for the initial iteration.
* `nr.employed`: Contains the largest absolute values among all numerical features and requires scaling/normalization.
* **Multicollinearity:** Strong correlations were detected between `euribor3m`, `emp.var.rate`, and `nr.employed`. For linear or regression models, only one of these features should be retained. However, since regression is only used as a baseline, I decided to keep all these variables because LightGBM and Random Forest are not sensitive to multicollinearity.

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
## Methodology Notes
- Data was split into Train (64%), Validation (16%), and Test (20%) sets with stratification on the target.
- Decision threshold and model selection were tuned exclusively on the Validation set.
- The Test set was used only once, for the final unbiased evaluation shown below.


---
## Model Performance & Results

After hyperparameter tuning and optimization, the models achieved the following results:

| Model Name | Train F1 | Val F1 | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Randomized Search LGBM** | **0.521** | **0.511** | **0.525** | **0.468** | **0.599** |
| Baseline LGBM | 0.569 | 0.505 | 0.520 | 0.464 | 0.593 |
| RandomForest | 0.517 | 0.509 | 0.524 | 0.454 | 0.620 |
| Hyperopt LGBM | 0.504 | 0.498 | 0.517 | 0.455 | 0.599 |
| Logistic Regression | 0.432 | 0.431 | 0.451 | 0.346 | 0.645 |
| KNN | 0.476 | 0.376 | 0.379 | 0.584 | 0.280 |

---

## Conclusion
The **Randomized Search LightGBM (LGBM)** model emerged as the top performer.

* **Highest Test F1 (0.525):** Demonstrates the best overall balance between Precision and Recall with the fewest cumulative errors.
* **Precision (~0.47):** For a bank, this implies that almost every second marketing call based on the model's prediction will be successful.
* **Generalization:** Unlike Baseline LGBM - which shows a large gap between Train (0.569) and Val/Test (~0.51-0.52), a sign of overfitting - Randomized Search LGBM shows minimal variation across Train (0.521), Val (0.511), and Test (0.525) F1-scores, indicating stronger generalization rather than a lucky test-set result.

### Future Work
To further improve model performance, future iterations will focus on advanced **Feature Engineering** to create new, high-quality domain-specific features.

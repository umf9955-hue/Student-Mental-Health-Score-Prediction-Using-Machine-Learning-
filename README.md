# Student-Mental-Health-Score-Prediction-Using-Machine-Learning-
This project investigates the relationship between students' daily habits (social media usage, AI tool usage, sleep duration, physical activity) and their mental health scores using regression modeling and machine learning techniques in Python.
![Department](https://img.shields.io/badge/Department-ECE-blue)
![Institution](https://img.shields.io/badge/Institute-PAF--IAST-green)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📌 Project Overview
Mental health and well-being are crucial aspects of a student's academic and personal life[cite: 3]. Daily habits—such as sleep duration, physical activity, and time spent on social media or AI tools—may significantly impact mental health outcomes[cite: 3]. 

---

## 👨‍🎓 Project Details
* **Author:** Umer Farooq[cite: 1]
* **Registration No:** B24F0418A1076[cite: 1]
* **Supervised by:** Dr. Abid Ali[cite: 1]
* **Course / Assignment:** Machine Learning - Assignment 1[cite: 1]
* **Department:** Department of Electrical and Computer Engineering[cite: 1]
* **Institution:** Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology (PAF-IAST)[cite: 1]
* **Date:** October 24, 2026[cite: 1]

---

## 🎯 Business Problem & Objective
Educational institutions require data-driven evidence to understand how everyday student habits correlate with well-being[cite: 3]. Because these relationships are often non-linear or multi-faceted, standard observation can miss key trends[cite: 3].

Using the **AI & Social Media Impact on Student Health and Grades** dataset[cite: 3, 4], this project formulates a **regression task**[cite: 3]:
* **Target Variable:** `Mental_Health_Score` (Continuous value)[cite: 3, 4]
* **Input Features:** Daily habit metrics (social media, AI tool usage, sleep, physical activity) and demographic variables[cite: 3, 4].

---

## 📊 Dataset Summary
* **Source:** Kaggle (`srisyra02/ai-and-social-media-impact-student-health-and-grades`)[cite: 3, 14]
* **Total Records:** 16,000 rows[cite: 4]
* **Total Original Features:** 10 columns (8 input features + ID + Target)[cite: 4]
* **Features Post-Preprocessing:** 14 encoded/engineered features[cite: 4]
* **Train / Test Split:** 80% Training (12,800 rows) / 20% Testing (3,200 rows)[cite: 4]

### Variable Statistics
| Feature | Type | Min | Max | Mean | Std | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | `int64` | 13.00 | 25.00 | 19.04 | 3.77 | -0.011 | -1.218 |
| **Daily_Social_Media_Hours** | `float64` | 0.00 | 14.00 | 4.54 | 2.40 | 0.157 | -0.348 |
| **Daily_AI_Tool_Usage_Hours** | `float64` | 0.00 | 9.50 | 2.59 | 1.68 | 0.338 | -0.369 |
| **Sleep_Hours** | `float64` | 2.00 | 11.15 | 6.55 | 1.27 | 0.015 | -0.005 |
| **Physical_Activity_Hours** | `float64` | 0.00 | 5.00 | 1.04 | 1.25 | 0.619 | -0.273 |
| **Mental_Health_Score** *(Target)* | `float64` | 32.56 | 91.76 | 72.49 | 9.24 | -0.683 | 0.244 |
| **Physical_Health_Score** | `float64` | 48.03 | 99.98 | 88.02 | 9.69 | -0.749 | -0.038 |

---

## ⚙️ Data Preprocessing & Feature Engineering
1. **Data Cleaning:** Verified zero missing values (`0`) and no duplicate records or Student IDs[cite: 7].
2. **Outlier Assessment:** Evaluated via IQR boundaries; flagged instances were retained to preserve natural variation[cite: 8].
3. **Categorical Encoding & Feature Creation:** Categorical variables (`Gender`, `Education_Level`) were one-hot encoded[cite: 6, 7]. Feature expansion produced 14 final input columns[cite: 4, 7]:
   * `Age`, `Daily_Social_Media_Hours`, `Daily_AI_Tool_Usage_Hours`, `Sleep_Hours`, `Physical_Activity_Hours`, `Physical_Health_Score`[cite: 7]
   * `AI_Share_of_Screen_Time`, `Heavy_Social_Media`, `Low_Sleep`, `Inactive`[cite: 7]
   * `Gender_Male`, `Gender_Non-Binary`, `Education_Level_High School`, `Education_Level_University`[cite: 7]

---

## 📈 Exploratory Data Analysis Key Findings
* **Strongest Negative Correlation:** Daily social media usage shows a clear inverse relationship with mental health scores ($r = -0.40$)[cite: 8, 9].
* **Positive Factors:** Higher sleep hours ($r = 0.31$), physical activity ($r = 0.25$), and physical health scores ($r = 0.25$) correlate positively with higher mental health scores[cite: 8].
* **Demographics:** Mental health distributions show virtually identical medians and ranges across genders and education levels (High School, College, University)[cite: 9].

---

## 🛠️ Models & Methodology
1. **Linear Regression:** Baseline parametric regression model[cite: 3].
2. **Ridge Regression:** L2 regularized model with hyperparameter tuning ($\alpha = 100$)[cite: 12].
3. **Lasso Regression:** L1 regularized model with feature selection capability ($\alpha = 0.01$)[cite: 12].
4. **PCA + Linear Regression:** Dimensionality reduction reaching 95% explained variance threshold at ~8 principal components[cite: 12].

---

## 🚀 Experimental Results & Model Comparison

### Performance Evaluation (Test Set)
| Model | $R^2$ | RMSE | MAE | MSE | MAPE (%) | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.2390 | 8.0128 | 6.3280 | 64.2045 | 9.2751 | 0.0776 |
| **Ridge Regression** | 0.2392 | 8.0115 | 6.3276 | 64.1835 | 9.2751 | 4.7751 |
| **Lasso Regression** | **0.2395** | **8.0098** | **6.3268** | **64.1576** | **9.2729** | 1.7399 |
| **PCA + Linear Reg.** | 0.2385 | 8.0154 | 6.3300 | 64.2469 | 9.2791 | **0.0618** |

### Top Standardized Feature Coefficients
| Feature | Coefficient Effect | Impact Direction |
| :--- | :--- | :--- |
| **Daily_Social_Media_Hours** | **-2.948** | Strong Negative |
| **Sleep_Hours** | **+1.802** | Strong Positive |
| **Physical_Activity_Hours** | **+1.496** | Moderate Positive |
| **Inactive** | -0.578 | Negative |
| **Daily_AI_Tool_Usage_Hours** | -0.557 | Negative |
| **Education_Level_University** | +0.479 | Slight Positive |

---

## 💡 Key Insights & Conclusion
* **Primary Drivers:** Excessive daily social media consumption is the single strongest negative predictor of student mental health scores, whereas adequate sleep and regular physical activity are the strongest positive predictors[cite: 11].
* **Generalization:** Training $R^2$ (0.2476) and test $R^2$ (0.2390) show a minimal generalization gap (0.0086), indicating no overfitting[cite: 10, 13].
* **Unexplained Variance:** ~76% of variance remains unexplained by daily habit metrics alone[cite: 13]. Future research should incorporate variables such as academic stress levels, financial stability, and personal support systems[cite: 13].

---

## 📚 References
1. World Health Organization, *"Adolescent mental health,"* WHO Fact Sheet[cite: 14].
2. Kaggle Dataset, *"AI & Social Media Impact: Student Health & Grades"* by srisyra02[cite: 14].
3. F. Pedregosa et al., *"Scikit-learn: Machine Learning in Python,"* JMLR, vol. 12, pp. 2825-2830, 2011[cite: 14].
4. C. R. Harris et al., *"Array programming with NumPy,"* Nature, vol. 585, pp. 357-362, 2020[cite: 14].
5. W. McKinney, *"Data Structures for Statistical Computing in Python,"* SciPy, 2010[cite: 14].
6. T. Hastie, R. Tibshirani, and J. Friedman, *"The Elements of Statistical Learning,"* 2nd ed. Springer, 2009[cite: 14].

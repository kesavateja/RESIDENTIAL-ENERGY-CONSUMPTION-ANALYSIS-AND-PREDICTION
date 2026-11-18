import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)



df = pd.read_csv("recs2015_public_v4.csv")

print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA TYPES")
print(df.dtypes)

print("\nSTATISTICAL DESCRIPTION")
print(df.describe())

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTOTAL MISSING VALUES")
print(df.isnull().sum().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

df = df.drop_duplicates()

df = df[df['BEDROOMS'] >= 0]

print("\nDATA SHAPE AFTER CLEANING")
print(df.shape)

q1 = df['TOTALBTU'].quantile(0.25)

q3 = df['TOTALBTU'].quantile(0.75)

IQR = q3 - q1

print("\nTOTALBTU IQR")
print(IQR)

lb = q1 - 1.5 * IQR

ub = q3 + 1.5 * IQR

print("Lower Bound:", lb)

print("Upper Bound:", ub)

outliers = ((df['TOTALBTU'] < lb) |
            (df['TOTALBTU'] > ub)).sum()

print("TOTALBTU OUTLIERS:", outliers)


q1_sqft = df['TOTSQFT_EN'].quantile(0.25)

q3_sqft = df['TOTSQFT_EN'].quantile(0.75)

IQR_sqft = q3_sqft - q1_sqft

print("\nTOTSQFT_EN IQR")
print(IQR_sqft)

lb_sqft = q1_sqft - 1.5 * IQR_sqft

ub_sqft = q3_sqft + 1.5 * IQR_sqft

print("Lower Bound:", lb_sqft)

print("Upper Bound:", ub_sqft)

outliers_sqft = (
    (df['TOTSQFT_EN'] < lb_sqft) |
    (df['TOTSQFT_EN'] > ub_sqft)
).sum()

print("TOTSQFT_EN OUTLIERS:", outliers_sqft)

plt.figure(figsize=(8, 5))

plt.boxplot(df['TOTALBTU'])

plt.title("Boxplot of TOTALBTU")

plt.ylabel("TOTALBTU")

plt.show()

plt.figure(figsize=(8, 5))

plt.boxplot(df['TOTSQFT_EN'])

plt.title("Boxplot of TOTSQFT_EN")

plt.ylabel("TOTSQFT_EN")

plt.show()

plt.figure(figsize=(8, 5))

plt.hist(df['TOTALBTU'], bins=20)

plt.title("Distribution of TOTALBTU")

plt.xlabel("TOTALBTU")

plt.ylabel("Frequency")

plt.show()

print("\nAIR CONDITIONING VALUE COUNTS")

print(df['AIRCOND'].value_counts())

df['AIRCOND'].value_counts().plot(kind='bar')

plt.title("AIRCOND Value Counts")

plt.xlabel("AIRCOND")

plt.ylabel("Count")

plt.show()

correlation_columns = [
    'TOTSQFT_EN',
    'TOTROOMS',
    'BEDROOMS',
    'HDD65',
    'CDD65',
    'NHSLDMEM',
    'MONEYPY',
    'YEARMADERANGE',
    'TOTALBTU',
    'AIRCOND'
]

plt.figure(figsize=(12, 8))

sns.heatmap(
    df[correlation_columns].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df['TOTSQFT_EN'],
    df['TOTALBTU'],
    color='blue'
)

plt.title("TOTSQFT_EN vs TOTALBTU")

plt.xlabel("TOTSQFT_EN")

plt.ylabel("TOTALBTU")

plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    df['HDD65'],
    df['TOTALBTU'],
    color='orange'
)

plt.title("HDD65 vs TOTALBTU")

plt.xlabel("HDD65")

plt.ylabel("TOTALBTU")

plt.show()

df_clean = df[
    (df['TOTALBTU'] >= lb) &
    (df['TOTALBTU'] <= ub)
].copy()

features = [
    'TOTSQFT_EN',
    'TOTROOMS',
    'BEDROOMS',
    'HDD65',
    'CDD65',
    'NHSLDMEM',
    'MONEYPY',
    'YEARMADERANGE'
]

x = df_clean[features]

y = df_clean['TOTALBTU']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


df_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print("\nMULTIPLE LINEAR REGRESSION RESULTS")

print(df_results.head(10))

MAE = mean_absolute_error(y_test, y_pred)

MSE = mean_squared_error(y_test, y_pred)

RMSE = np.sqrt(MSE)

r2 = r2_score(y_test, y_pred)

print("\nMULTIPLE LINEAR REGRESSION EVALUATION")

print("MAE:", MAE)

print("MSE:", MSE)

print("RMSE:", RMSE)

print("R2 SCORE:", r2)

plt.figure(figsize=(12, 5))

plt.scatter(
    y_test,
    y_pred,
    color='blue',
    label='Actual vs Predicted'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    label='Ideal Line'
)

plt.title("Multiple Linear Regression: Actual vs Predicted")

plt.xlabel("Actual TOTALBTU")

plt.ylabel("Predicted TOTALBTU")

plt.legend()

plt.show()

poly = PolynomialFeatures(degree=2)

x_train_trans = poly.fit_transform(x_train)

x_test_trans = poly.transform(x_test)

poly_model = LinearRegression()

poly_model.fit(x_train_trans, y_train)

y_pred_poly = poly_model.predict(x_test_trans)


df_results_poly = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_poly
})

print("\nPOLYNOMIAL REGRESSION RESULTS")

print(df_results_poly.head(10))

MAE_poly = mean_absolute_error(y_test, y_pred_poly)

MSE_poly = mean_squared_error(y_test, y_pred_poly)

RMSE_poly = np.sqrt(MSE_poly)

r2_poly = r2_score(y_test, y_pred_poly)

print("\nPOLYNOMIAL REGRESSION EVALUATION")

print("MAE:", MAE_poly)

print("MSE:", MSE_poly)

print("RMSE:", RMSE_poly)

print("R2 SCORE:", r2_poly)

plt.figure(figsize=(12, 5))

plt.scatter(
    y_test,
    y_pred_poly,
    color='green',
    label='Actual vs Predicted'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    label='Ideal Line'
)

plt.title("Polynomial Regression: Actual vs Predicted")

plt.xlabel("Actual TOTALBTU")

plt.ylabel("Predicted TOTALBTU")

plt.legend()

plt.show()

df_class = df[
    (df['TOTALBTU'] >= lb) &
    (df['TOTALBTU'] <= ub) &
    (df['TOTSQFT_EN'] >= lb_sqft) &
    (df['TOTSQFT_EN'] <= ub_sqft)
].copy()

x = df_class[features]

y = df_class['AIRCOND']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

lr = LogisticRegression(max_iter=2000)

lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

y_prob_lr = lr.predict_proba(x_test)

print("\nLOGISTIC REGRESSION PREDICTION PROBABILITIES")

print(y_prob_lr[:10])

df_results_lr = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_lr
})

print("\nLOGISTIC REGRESSION RESULTS")

print(df_results_lr.head(10))

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("\nLOGISTIC REGRESSION ACCURACY")

print(accuracy_lr)

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nCONFUSION MATRIX")

print(cm_lr)

print("\nCLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_lr))

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - Logistic Regression")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train_scaled, y_train)

y_pred_knn = knn.predict(x_test_scaled)

df_results_knn = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_knn
})

print("\nKNN CLASSIFICATION RESULTS")

print(df_results_knn.head(10))

accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("\nKNN ACCURACY")

print(accuracy_knn)

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\nKNN CONFUSION MATRIX")

print(cm_knn)

print("\nKNN CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred_knn))

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Confusion Matrix - KNN")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

comparison_reg = pd.DataFrame({

    'Model': [
        'Multiple Linear Regression',
        'Polynomial Regression'
    ],

    'MAE': [
        MAE,
        MAE_poly
    ],

    'MSE': [
        MSE,
        MSE_poly
    ],

    'RMSE': [
        RMSE,
        RMSE_poly
    ],

    'R2_Score': [
        r2,
        r2_poly
    ]

})

print("\nREGRESSION MODEL COMPARISON")

print(comparison_reg)

comparison_clf = pd.DataFrame({

    'Model': [
        'Logistic Regression',
        'KNN Classifier'
    ],

    'Accuracy': [
        accuracy_lr,
        accuracy_knn
    ]

})

print("\nCLASSIFICATION MODEL COMPARISON")

print(comparison_clf)

print("\nDataset Shape:", df.shape)

print("\nRegression Models:")

print(comparison_reg)

print("\nClassification Models:")

print(comparison_clf)

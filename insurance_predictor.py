"""
MEDICAL INSURANCE COST PREDICTOR
Uses Random Forest Regression to estimate medical insurance costs.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Load dataset
DATA_FILE = "C:/Users/Lenovo/Desktop/New folder/all HW projects/insurance dataset.xlsx"
df = pd.read_excel(DATA_FILE)
print("=" * 60)
print("             MEDICAL INSURANCE DATASET")
print("=" * 60)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nDataset information:")
df.info()
print("\nMissing values:")
print(df.isnull().sum())
print("\nStatistical summary:")
print(df.describe())

# Exploratory Data Analysis
print("\n" + "=" * 60)
print("             EXPLORATORY DATA ANALYSIS")
print("=" * 60)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="charges", kde=True)
plt.title("Distribution of Medical Insurance Charges")
plt.xlabel("Insurance Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="charges")
plt.title("Age vs Medical Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="bmi", y="charges")
plt.title("BMI vs Medical Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="smoker", y="charges")
plt.title("Smoking Status vs Medical Insurance Charges")
plt.xlabel("Smoking Status")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="children", y="charges")
plt.title("Number of Children vs Medical Insurance Charges")
plt.xlabel("Number of Children")
plt.ylabel("Insurance Charges")
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(9, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Insurance Dataset")
plt.tight_layout()
plt.show()

# Prepare data for Machine Learning
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].map({
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
})

X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining data:", len(X_train), "records")
print("Testing data:", len(X_test), "records")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Model training completed successfully!")

# Model Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "=" * 60)
print("             MODEL EVALUATION")
print("=" * 60)
print(f"\nR² Score: {r2:.4f} ({r2:.2%})")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Factors Affecting Medical Insurance Cost")
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("Actual vs Predicted Medical Insurance Charges")
plt.tight_layout()
plt.show()

# User Input
print("\n" + "=" * 60)
print("        WELCOME TO THE MEDICAL INSURANCE PREDICTOR")
print("=" * 60)
print("\nPlease enter your details below.\n")

name = input("Enter your name: ").strip()

while True:
    try:
        age = int(input("Enter your age: "))
        if age > 0:
            break
        print("Please enter a valid age.")
    except ValueError:
        print("Please enter a number.")

while True:
    sex_input = input("Enter your sex (male/female): ").strip().lower()
    if sex_input == "male":
        sex = 0
        break
    elif sex_input == "female":
        sex = 1
        break
    else:
        print("Please enter male or female.")

while True:
    try:
        height_cm = float(input("Enter your height in cm (e.g. 170): "))
        if height_cm > 0:
            break
        print("Height must be greater than 0.")
    except ValueError:
        print("Please enter a valid number.")

while True:
    try:
        weight = float(input("Enter your weight in kg (e.g. 65): "))
        if weight > 0:
            break
        print("Weight must be greater than 0.")
    except ValueError:
        print("Please enter a valid number.")

height_m = height_cm / 100
bmi = weight / (height_m ** 2)

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal weight"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obesity range"

while True:
    try:
        children = int(input("Enter number of children/dependents: "))
        if children >= 0:
            break
        print("Number cannot be negative.")
    except ValueError:
        print("Please enter a whole number.")

while True:
    smoker_input = input("Do you smoke? (yes/no): ").strip().lower()
    if smoker_input == "yes":
        smoker = 1
        break
    elif smoker_input == "no":
        smoker = 0
        break
    else:
        print("Please enter yes or no.")

print("\nRegions available:")
print("northeast")
print("northwest")
print("southeast")
print("southwest")

region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

while True:
    region_input = input("Enter your region: ").strip().lower()
    if region_input in region_map:
        region = region_map[region_input]
        break
    else:
        print("Please enter one of the four regions listed above.")

# Prediction
new_person = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

predicted_yearly = model.predict(new_person)[0]
predicted_monthly = predicted_yearly / 12

print("\n" + "=" * 60)
print("              PREDICTION RESULT")
print("=" * 60)
print(f"\nHello {name}, based on the information you provided:")
print(f"Your BMI is: {bmi:.1f} ({bmi_category})")
print(f"Estimated Yearly Insurance Cost: ${predicted_yearly:,.2f}")
print(f"Estimated Monthly Insurance Cost: ${predicted_monthly:,.2f}")
print("\nNote: This is a machine-learning estimate based")
print("on patterns in the provided dataset. It is not")
print("a guaranteed insurance quote.")
print("=" * 60)
print("       THANK YOU FOR USING THE PREDICTOR!")
print("=" * 60)

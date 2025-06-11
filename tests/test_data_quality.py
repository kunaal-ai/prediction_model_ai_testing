import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from cerberus import Validator
import json

# Initialize a dictionary to hold the summary report
summary_report = {}

print(" \n Step 1 - set up and data loading .....")
# load dataset
try:
    df = pd.read_csv("data/raw/diabetes.csv")
    print("✔️ Data loaded successfully")
    print("1...5 rows")
    print(df.head())
    summary_report['data_loading'] = {
        'status': 'success',
        'rows':df.shape[0],
        'columns': df.shape[1]
    }
except Exception as e:
    print(f"Error loading data: {e}")
    summary_report['data_loading'] = {'status': 'error', 'message': str(e)}
    exit()
print("✅ Step 1 - set up and data loading completed")

# data profiling
print(" \n Step 2 - data profiling .....")
try:
    profile = ProfileReport(df, title = "Diabetes Data profile", minimal=True)
    profile.to_file("reports/diabetes_profile.html")
    print("✔️ Data profiling report saved in reports/diabetes_profile.html")
    summary_report['data_profiling'] = {
        'status': 'success',
        'report_path': 'reports/diabetes_profile.html'
    }
except Exception as e:
    print(f" Error profiling data: {e}")
    summary_report['data_profiling'] = {'status': 'error', 'message': str(e)}
    exit()
print("✅  Step 2 - data profiling completed")


# statistical Analysis
print(" \n Step 3 - statistical Analysis .....")
desc_stats = df.describe()
print(desc_stats)
summary_report['statistical_analysis'] = {
    'status': 'success',
    'description_stats': desc_stats.to_dict()
}

# checking missing values
print("missing values")
missing_values = df.isnull().sum()
print(f"Missing values: \n {missing_values} ")
summary_report['missing_values']= missing_values.to_dict()

#checking duplicate records
print("duplicate records")
duplicate_count = df.duplicated().sum()
print(f"duplicate total records: {duplicate_count}")
summary_report['duplicate_records'] = int(duplicate_count)


# following col != 0
summary_report['impossible_zeros'] = {}
zero_value_col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_value_col:
    zero_count = (df[col] == 0).sum()
    print(f"Number of zero values in '{col}': {zero_count} ({(zero_count / len(df)) * 100:.2f}%)")
    summary_report['impossible_zeros'][col] = int(zero_count)


# outlier detection - Interquartile Range (IQR) Method
summary_report['outliers'] = {}
outlier_col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in outlier_col:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    print(f"Number of outliers in '{col}': {outlier_count} ")
    summary_report['outliers'][col] = int(outlier_count)

# visualize Data Distribution- histograms for each feature
print("visualizing data distribution")
df.hist(bins=20, figsize=(2, 15))
plt.suptitle("Histograms of each feature")
plt.savefig("reports/histograms.png")
plt.close()

plt.figure(figsize=(20, 15))
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.savefig("reports/boxplots.png")
plt.close()
print("Saved boxplots to 'reports/boxplots.png'")

# correlation analysis
print("Correlation analysis")
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.savefig("reports/correlation_matrix.png")
plt.close()
print("Saved correlation matrix plot to 'reports/correlation_matrix.png'")
summary_report['correlation_matrix'] = correlation_matrix.to_dict()

# Rule Based validation
print("Rule based validation with Cerberus")
schema = {
    'Pregnancies': {'type': 'number', 'min': 0},
    'Glucose': {'type': 'number', 'min': 1, 'nullable': True}, # Allowing null after replacement
    'BloodPressure': {'type': 'number', 'min': 1, 'nullable': True},
    'SkinThickness': {'type': 'number', 'min': 1, 'nullable': True},
    'Insulin': {'type': 'number', 'min': 1, 'nullable': True},
    'BMI': {'type': 'float', 'min': 1.0, 'nullable': True},
    'DiabetesPedigreeFunction': {'type': 'float', 'min': 0.0},
    'Age': {'type': 'number', 'min': 21},
    'Outcome': {'type': 'number', 'allowed': [0, 1]}
}
v = Validator(schema)
df_for_validation = df.copy()
df_for_validation[zero_value_col] = df_for_validation[zero_value_col].replace(0, np.nan) # nan - not a number

# Validate the entire DataFrame
validation_errors = []
for index, row in df_for_validation.iterrows():
    if not v.validate(row.to_dict()):
        validation_errors.append({'row_index': index, 'errors': v.errors})

if validation_errors:
    print(f"Found {len(validation_errors)} rows with validation errors.")
    print("First 5 errors:")
    for error in validation_errors[:5]:
        print(error)
else:
    print("No validation errors found in the entire dataset.")

summary_report['validation_errors'] = {
    'error_count': len(validation_errors),
    'errors': validation_errors
}

# --- 5. Generate Summary Report ---
print("\n--- Step 5: Generating JSON Summary Report ---")
try:
    with open('reports/data_quality_report.json', 'w') as f:
        json.dump(summary_report, f, indent=4)
    print("Successfully generated 'reports/data_quality_report.json'")
except Exception as e:
    print(f"Error generating JSON report: {e}")

print("\n--- Data Quality Testing Complete ---")

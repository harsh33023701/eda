
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Clean graph style
plt.style.use('ggplot')


# LOAD DATASET

file_path = "Summer_Meal_Programs_-_All_Summer_Sites__-_Contact_and_Program_Participation_Information_-_Program_Period_2019 (2).csv"

df = pd.read_csv(file_path)

print("\n DATA LOADED")
print(df.head())


#BASIC EDA

print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nMissing Values:\n", df.isnull().sum())


#DATA CLEANING

df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)


#  DATE HANDLING

date_cols = [col for col in df.columns if "date" in col.lower()]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering
if len(date_cols) >= 1:
    df['Month'] = df[date_cols[0]].dt.month

if len(date_cols) >= 2:
    df['Duration'] = (df[date_cols[1]] - df[date_cols[0]]).dt.days


# NUMERIC DATA

num_df = df.select_dtypes(include=np.number)


#CLEAN GRAPHS


# Histogram (only first 3 columns)
if not num_df.empty:
    for col in num_df.columns[:3]:
        plt.figure()
        plt.hist(num_df[col].dropna(), bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# Boxplot
if not num_df.empty:
    plt.figure()
    num_df.iloc[:, :3].plot(kind='box')
    plt.title("Boxplot (Outliers)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Correlation Heatmap
if not num_df.empty:
    corr = num_df.corr()
    print("\nCorrelation:\n", corr)

    plt.figure()
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

#Bar Chart (Top States)
state_col = None
for col in df.columns:
    if "state" in col.lower():
        state_col = col
        break

if state_col:
    top_states = df[state_col].value_counts().head(10)

    plt.figure()
    top_states.plot(kind='bar')
    plt.title("Top 10 States by Sites")
    plt.xlabel("State")
    plt.ylabel("Number of Sites")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Pie Chart (Top Sponsors)
sponsor_col = None
for col in df.columns:
    if "sponsor" in col.lower():
        sponsor_col = col
        break

if sponsor_col:
    top_sponsors = df[sponsor_col].value_counts().head(5)

    plt.figure()
    top_sponsors.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Top Sponsors Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# Line Chart (Month Trend)
if 'Month' in df.columns:
    month_data = df['Month'].value_counts().sort_index()

    plt.figure()
    month_data.plot(marker='o')
    plt.title("Programs Over Months")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.grid()
    plt.tight_layout()
    plt.show()


# REGRESSION + R2

if 'Month' in df.columns and 'Duration' in df.columns:
    X = df[['Month']].fillna(0)
    y = df['Duration'].fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    print("\nR2 Score:", r2_score(y, y_pred))

    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, y_pred)
    plt.title("Regression Analysis")
    plt.xlabel("Month")
    plt.ylabel("Duration")
    plt.tight_layout()
    plt.show()


# T-TEST

cols = num_df.columns

if len(cols) >= 2:
    t, p = stats.ttest_ind(num_df[cols[0]].dropna(), num_df[cols[1]].dropna())

    print("\nT-Test Results:")
    print("T-value:", t)
    print("P-value:", p)


# Z-TEST

if len(cols) >= 1:
    sample = num_df[cols[0]].dropna()

    mean = np.mean(sample)
    std = np.std(sample)
    n = len(sample)

    z = (mean - 0) / (std / np.sqrt(n))

    print("\nZ-Test Value:", z)

# ==============================
# 11. SAVE CLEANED DATA
# ==============================
df.to_csv("Cleaned_Output.csv", index=False)

print("\nDONE! Cleaned file saved as Cleaned_Output.csv")

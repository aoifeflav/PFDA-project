#This s a remporary file usd for rough work for my final project
# Author: Aoife Flavin

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv(
    'sherkin_island_weather.csv', skiprows=17, dtype=str, low_memory=False)

print(df.columns)
#https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options



df.columns = [
    "Date and Time (UTC)", "Indicator 1", "Precipitation Amount (mm)", "Indicator 2",
    "Air Temperature (C)", "Indicator 3", "Wet Bulb Temperature (C)", 
    "Dew Point Temperature (C)", "Vapour Pressure (hPa)", 
    "Relative Humidity (%)", "Mean Sea Level Pressure (hPa)", 
    "Indicator 4", "Mean Wind Speed (knot)", "Indicator 5", 
    "Predominant Wind Direction (degree)"
]
df = df.iloc[:, :15]  # Keep only the first 15 columns
new_column_names = df.columns




numeric_columns = [
    "Precipitation Amount (mm)",
    "Air Temperature (C)",
    "Wet Bulb Temperature (C)",
    "Dew Point Temperature (C)",
    "Vapour Pressure (hPa)",
    "Relative Humidity (%)",
    "Mean Sea Level Pressure (hPa)",
    "Mean Wind Speed (knot)",
    "Predominant Wind Direction (degree)"
]

# Convert the specified numeric columns to float, coercing invalid values to NaN
for col in numeric_columns:
    if col in df.columns:  # Check if the column exists after renaming
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Invalid -> NaN



# Parse "Date and Time (UTC)" with the specified format
df["Date and Time (UTC)"] = pd.to_datetime(
    df["Date and Time (UTC)"], 
    format="%d-%b-%Y %H:%M",  # This matches the '30-apr-2004 06:00' format
    errors="coerce"  # Invalid -> NaT
)

# Remove rows with invalid dates
df.dropna(subset=["Date and Time (UTC)"], inplace=True)




# Drop columns with all null values
df.dropna(axis=1, how='all', inplace=True)

# Drop rows with any null values
df.dropna(axis=0, inplace=True)

# Convert the "Date and Time (UTC)" column to datetime format
df["Date and Time (UTC)"] = pd.to_datetime(df["Date and Time (UTC)"])

# Display cleaned data
print(df.head())




#Find out what data types each variable is
print(df.info())



#convert inalid entries to nan
df["Air Temperature (C)"] = pd.to_numeric(df["Air Temperature (C)"], errors="coerce")
df["Relative Humidity (%)"] = pd.to_numeric(df["Relative Humidity (%)"], errors="coerce")
df["Mean Wind Speed (knot)"] = pd.to_numeric(df["Mean Wind Speed (knot)"], errors="coerce")

df.dropna(subset=["Air Temperature (C)"], inplace=True)
df.dropna(subset=["Relative Humidity (%)"], inplace=True)
df.dropna(subset=["Mean Wind Speed (knot)"], inplace=True)



# Filter data for winter (Nov-Jan) and summer (May-Jul)
winter_months = [11, 12, 1]
summer_months = [5, 6, 7]

# Just the month
df["Month"] = df["Date and Time (UTC)"].dt.month

# filter winter and summer data
winter_data = df[df["Month"].isin(winter_months)]
summer_data = df[df["Month"].isin(summer_months)]

# create function for summary stats
def compute_summary_statistics(data, variable_name):
    stats = {
        "Mean": np.mean(data),
        "Median": np.median(data),
        "Variance": np.var(data),
        "Standard Deviation": np.std(data)
    }
    return stats

# calculate winter stats
winter_stats = {
    "Temperature (C)": compute_summary_statistics(winter_data["Air Temperature (C)"], "Temperature"),
    "Humidity (%)": compute_summary_statistics(winter_data["Relative Humidity (%)"], "Humidity"),
    "Wind Speed (knot)": compute_summary_statistics(winter_data["Mean Wind Speed (knot)"], "Wind Speed"),
}

# calculate summer stats
summer_stats = {
    "Temperature (C)": compute_summary_statistics(summer_data["Air Temperature (C)"], "Temperature"),
    "Humidity (%)": compute_summary_statistics(summer_data["Relative Humidity (%)"], "Humidity"),
    "Wind Speed (knot)": compute_summary_statistics(summer_data["Mean Wind Speed (knot)"], "Wind Speed"),
}

# Display results
print("Winter Summary Statistics:")
for variable, stats in winter_stats.items():
    print(f"{variable}: {stats}")

print("\nSummer Summary Statistics:")
for variable, stats in summer_stats.items():
    print(f"{variable}: {stats}")




# Filter data for the specific date (19th November 2024)
specific_date = "2024-11-19"
filtered_data = df[df["Date and Time (UTC)"].dt.date == pd.to_datetime(specific_date).date()]

# Plot Wind Speed over time
plt.figure(figsize=(10, 6))
plt.plot(filtered_data["Date and Time (UTC)"], filtered_data["Mean Wind Speed (knot)"], marker='o', color='blue', label="Wind Speed (knot)")
plt.xlabel("Time (UTC)")
plt.ylabel("Wind Speed (knot)")
plt.title(f"Fluctuation of Wind Speed on {specific_date}")
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()



# date range
start_date = "2024-11-23"
end_date = "2024-11-30"

# Filter data
filtered_data = df[(df["Date and Time (UTC)"] >= pd.to_datetime(start_date)) & 
                   (df["Date and Time (UTC)"] <= pd.to_datetime(end_date))]

# Plot Wind Speed over week
plt.figure(figsize=(12, 6))
plt.plot(filtered_data["Date and Time (UTC)"], filtered_data["Mean Wind Speed (knot)"], marker='o', color='blue', label="Wind Speed (knot)")
plt.xlabel("Date and Time (UTC)")
plt.ylabel("Wind Speed (knot)")
plt.title(f"Fluctuation of Wind Speed from {start_date} to {end_date}")
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# date for grouping
df["Date"] = df["Date and Time (UTC)"].dt.date

#group by date and calculate daily mean
daily_data = df.groupby("Date").mean()

# Plot 
plt.figure(figsize=(12, 6))
plt.plot(daily_data.index, daily_data["Mean Wind Speed (knot)"], marker='o', color='orange', label="Mean Wind Speed (knot)")
plt.xlabel("Date")
plt.ylabel("Mean Wind Speed (knot)")
plt.title("Daily Trend of Wind Speed")
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
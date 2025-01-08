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

# convert to float, coercing invalid values to NaN
for col in numeric_columns:
    if col in df.columns:  
        df[col] = pd.to_numeric(df[col], errors="coerce")  # convert invalid columns to Nan



# date time formatting
df["Date and Time (UTC)"] = pd.to_datetime(
    df["Date and Time (UTC)"], 
    format="%d-%b-%Y %H:%M",  
    errors="coerce"  
)

# get rid of rows with invalid dates
df.dropna(subset=["Date and Time (UTC)"], inplace=True)




df.dropna(how="all", inplace=True)




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


# Define date range
start_date = pd.to_datetime("2023-11-30").date()
end_date = pd.to_datetime("2024-11-30").date()

# filter using date column
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Get daily max wind speed
daily_max_wind = filtered_df.groupby("Date")["Mean Wind Speed (knot)"].max().reset_index()
daily_max_wind.rename(columns={"Mean Wind Speed (knot)": "Daily Max Wind Speed (knot)"}, inplace=True)

#Calculate 30 day rolling average 
daily_max_wind["Rolling Avg Max Wind Speed (knot)"] = daily_max_wind["Daily Max Wind Speed (knot)"].rolling(window=30).mean()

# Plot 
plt.figure(figsize=(12, 6))
plt.plot(daily_max_wind["Date"], daily_max_wind["Rolling Avg Max Wind Speed (knot)"], color='blue', label="30-Day Rolling Avg Max Wind Speed")
plt.xlabel("Date")
plt.ylabel("Wind Speed (knot)")
plt.title(f"30 Day Rolling Average of Daily Max Wind Speed ({start_date} to {end_date})")
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
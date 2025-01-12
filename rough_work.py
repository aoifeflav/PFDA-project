#This s a remporary file usd for rough work for my final project
# Author: Aoife Flavin

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from matplotlib.dates import DateFormatter

from datetime import timedelta



# Function to load and clean a dataset
def load_and_clean_data(file_path, skiprows, column_names, numeric_columns):
    df = pd.read_csv(file_path, skiprows=skiprows, dtype=str, low_memory=False)
    
    df.columns = column_names
    
    # keep only relevant columns
    df = df.iloc[:, :len(column_names)]
    
    # Convert to float
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # datetime
    df["Date and Time (UTC)"] = pd.to_datetime(
        df["Date and Time (UTC)"], 
        format="%d-%b-%Y %H:%M", 
        errors="coerce"
    )
    
    # drop rows with invalid dates
    df.dropna(subset=["Date and Time (UTC)"], inplace=True)
    
    # Drop nan values
    df.dropna(how="all", inplace=True)
    
    return df

# Sherkin Island 
sherkin_column_names = [
    "Date and Time (UTC)", "Indicator 1", "Precipitation Amount (mm)", "Indicator 2",
    "Air Temperature (C)", "Indicator 3", "Wet Bulb Temperature (C)", 
    "Dew Point Temperature (C)", "Vapour Pressure (hPa)", 
    "Relative Humidity (%)", "Mean Sea Level Pressure (hPa)", 
    "Indicator 4", "Mean Wind Speed (knot)", "Indicator 5", 
    "Predominant Wind Direction (degree)"
]
sherkin_numeric_columns = [
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
df = load_and_clean_data('sherkin_island_weather.csv', skiprows=17, 
                         column_names=sherkin_column_names, 
                         numeric_columns=sherkin_numeric_columns)

# Cork Airport 
cork_column_names = [
    "Date and Time (UTC)", "Indicator 1", "Precipitation Amount (mm)", "Indicator 2",
    "Air Temperature (C)", "Indicator 3", "Wet Bulb Temperature (C)", 
    "Dew Point Temperature (C)", "Vapour Pressure (hPa)", 
    "Relative Humidity (%)", "Mean Sea Level Pressure (hPa)", 
    "Indicator 4", "Mean Wind Speed (knot)", "Indicator 5", 
    "Predominant Wind Direction (degree)", "Synop code for Present Weather", "Synop code for Past Weather", 
    "Sunshine duration (hours)", "Visibility (m)", "Cloud height (100's of ft)", "Cloud amount"
]
cork_numeric_columns = [
    "Precipitation Amount (mm)",
    "Air Temperature (C)",
    "Wet Bulb Temperature (C)",
    "Dew Point Temperature (C)",
    "Vapour Pressure (hPa)",
    "Relative Humidity (%)",
    "Mean Sea Level Pressure (hPa)",
    "Mean Wind Speed (knot)",
    "Predominant Wind Direction (degree)"
    "Sunshine duration (hours)"
    "Visibility (m)"
    "Cloud height (100's of ft)"
    "Cloud amount"
]
ca_df = load_and_clean_data('cork_airport_weather.csv', skiprows=23, 
                            column_names=cork_column_names, 
                            numeric_columns=cork_numeric_columns)

# Moore Park 
moore_column_names = [
    "Date and Time (UTC)", "Indicator 1", "Precipitation Amount (mm)", "Indicator 2",
    "Air Temperature (C)", "Indicator 3", "Wet Bulb Temperature (C)", 
    "Dew Point Temperature (C)", "Vapour Pressure (hPa)", 
    "Relative Humidity (%)", "Mean Sea Level Pressure (hPa)", 
    "Indicator 4", "Mean Wind Speed (knot)", "Indicator 5", 
    "Predominant Wind Direction (degree)"
]
moore_numeric_columns = [
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
mp_df = load_and_clean_data('moore_park_weather.csv', skiprows=17, 
                            column_names=moore_column_names, 
                            numeric_columns=moore_numeric_columns)




# Filter data from 2010 onwards
# Sherkin Island
df_filtered = df[df["Date and Time (UTC)"].dt.year >= 2010][["Date and Time (UTC)", "Mean Wind Speed (knot)"]]

# Cork Airport
ca_df_filtered = ca_df[ca_df["Date and Time (UTC)"].dt.year >= 2010][["Date and Time (UTC)", "Mean Wind Speed (knot)"]]

# Moore Park
mp_df_filtered = mp_df[mp_df["Date and Time (UTC)"].dt.year >= 2010][["Date and Time (UTC)", "Mean Wind Speed (knot)"]]


sherkin_stats = {
    "mean": df_filtered["Mean Wind Speed (knot)"].mean(),
    "median": df_filtered["Mean Wind Speed (knot)"].median(),
    "std_dev": df_filtered["Mean Wind Speed (knot)"].std()
}

cork_stats = {
    "mean": ca_df_filtered["Mean Wind Speed (knot)"].mean(),
    "median": ca_df_filtered["Mean Wind Speed (knot)"].median(),
    "std_dev": ca_df_filtered["Mean Wind Speed (knot)"].std()
}

moore_stats = {
    "mean": mp_df_filtered["Mean Wind Speed (knot)"].mean(),
    "median": mp_df_filtered["Mean Wind Speed (knot)"].median(),
    "std_dev": mp_df_filtered["Mean Wind Speed (knot)"].std()
}

print(f'Sherkin Island Stats {sherkin_stats}')
print(f'Cork Airport Stats {cork_stats}')
print(f'Moore Park Stats {moore_stats}')




# Plot wind speed data
plt.figure(figsize=(12, 6))

# Sherkin Island
plt.hist(df_filtered["Mean Wind Speed (knot)"], bins=20, alpha=0.5, label="Sherkin Island")

# Cork Airport
plt.hist(ca_df_filtered["Mean Wind Speed (knot)"], bins=20, alpha=0.5, label="Cork Airport")

# Moore Park
plt.hist(mp_df_filtered["Mean Wind Speed (knot)"], bins=20, alpha=0.5, label="Moore Park")

# design
plt.title("Wind Speed Distribution")
plt.xlabel("Mean Wind Speed (knot)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()





# function to calculate stats
def calculate_statistics(df, location_name):
    stats = {
        "Location": location_name,
        "Min": df["Mean Wind Speed (knot)"].min(),
        "Max": df["Mean Wind Speed (knot)"].max(),
        "IQR": df["Mean Wind Speed (knot)"].quantile(0.75) - df["Mean Wind Speed (knot)"].quantile(0.25),
    }
    return stats

# by location
sherkin_stats = calculate_statistics(df_filtered, "Sherkin Island")
cork_stats = calculate_statistics(ca_df_filtered, "Cork Airport")
moore_stats = calculate_statistics(mp_df_filtered, "Moore Park")

# results
stats_summary = pd.DataFrame([sherkin_stats, cork_stats, moore_stats])
print(stats_summary)



# Make a single dataframe
df_filtered["Location"] = "Sherkin Island"
ca_df_filtered["Location"] = "Cork Airport"
mp_df_filtered["Location"] = "Moore Park"

combined_df = pd.concat([df_filtered, ca_df_filtered, mp_df_filtered])

# boxplot for wind speed across locations
plt.figure(figsize=(10, 6))
sns.boxplot(data=combined_df, x="Location", y="Mean Wind Speed (knot)", palette="Set2")
plt.title("Wind Speed Comparison Across Cork", fontsize=14)
plt.xlabel("Location", fontsize=12)
plt.ylabel("Mean Wind Speed (knot)", fontsize=12)
plt.tight_layout()
plt.show()



#Create a column for date and month
df["Month"] = df["Date and Time (UTC)"].dt.month
df["Date"] = df["Date and Time (UTC)"].dt.date




# Filter data for winter (Nov-Feb) and summer (May-Aug)
winter_months = [11, 12, 1, 2]
summer_months = [5, 6, 7, 8]


# filter winter and summer data
winter_data = df[df["Month"].isin(winter_months)]
summer_data = df[df["Month"].isin(summer_months)]

# create function for summary stats
def compute_summary_statistics(data, variable_name):
    stats = {
        "Mean": np.mean(data),
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





# dates
dates = ["2024-11-19", "2024-08-19", "2024-05-19", "2024-02-10"]

# seaborn style
sns.set_theme(style="whitegrid")

# 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

# For loop over dates
for i, specific_date in enumerate(dates):
    # Filter data for the specific date
    filtered_data = df[df["Date and Time (UTC)"].dt.date == pd.to_datetime(specific_date).date()]
    
    # use seaborn to plot windspeed
    sns.lineplot(
        x="Date and Time (UTC)", 
        y="Mean Wind Speed (knot)", 
        data=filtered_data, 
        marker='o', 
        color='royalblue', 
        ax=axes[i]
    )
    
    # max wind speed label
    if not filtered_data.empty:
        max_speed = filtered_data["Mean Wind Speed (knot)"].max()
        max_time = filtered_data[filtered_data["Mean Wind Speed (knot)"] == max_speed]["Date and Time (UTC)"].iloc[0]
        axes[i].scatter(max_time, max_speed, color='red', s=50, label=f"Max: {max_speed:.2f}")
        axes[i].text(max_time, max_speed + 1, f"{max_speed:.2f}", color='red', fontsize=9)
    
    # titles & labels
    axes[i].set_title(f"Wind Speed on {specific_date}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Time (UTC)", fontsize=10)
    axes[i].xaxis.set_major_formatter(DateFormatter("%H:%M"))
    axes[i].tick_params(axis="x", rotation=45)
    axes[i].legend(fontsize=8)

# y label
fig.supylabel("Wind Speed (knot)", fontsize=12)


plt.tight_layout()
plt.show()



# date range
start_date = "2024-11-23"
end_date = "2024-11-30"

# filter data
filtered_data = df[(df["Date and Time (UTC)"] >= pd.to_datetime(start_date)) & 
                   (df["Date and Time (UTC)"] <= pd.to_datetime(end_date))]


sns.set_theme(style="whitegrid")

# plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=filtered_data["Date and Time (UTC)"], 
             y=filtered_data["Mean Wind Speed (knot)"], 
             marker='o', 
             color='royalblue')

# highlight max point
if not filtered_data.empty:
    max_speed = filtered_data["Mean Wind Speed (knot)"].max()
    max_time = filtered_data[filtered_data["Mean Wind Speed (knot)"] == max_speed]["Date and Time (UTC)"].iloc[0]
    
    plt.scatter(max_time, max_speed, color='red', s=50, label=f"Max: {max_speed:.2f} knot")
    plt.text(max_time, max_speed + 1, f"{max_speed:.2f}", color='red', fontsize=9)


plt.title(f"Fluctuation of Wind Speed from {start_date} to {end_date}", fontsize=14, fontweight='bold')
plt.xlabel("Date and Time (UTC)", fontsize=12)
plt.ylabel("Wind Speed (knot)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(fontsize=10)
plt.grid()

# format date
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

plt.tight_layout()
plt.show()



#datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Define date range for the past 3 years (use datetime, not date)
end_date = pd.to_datetime("2023-11-30")  # Removed .date()
start_date = end_date - timedelta(days=365 * 3)

# Filter data for the last 3 years
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
filtered_df['Year'] = filtered_df['Date'].dt.year
yearly_data = {year: filtered_df[filtered_df['Year'] == year] for year in range(end_date.year - 2, end_date.year + 1)}

# Initialize plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle("30-Day Rolling Average of Daily Max Wind Speed Over 3 Years", fontsize=16, weight='bold')

colors = ['blue', 'green', 'orange']

for i, (year, data) in enumerate(yearly_data.items()):
    # Calculate daily max wind speed
    daily_max_wind = data.groupby("Date")["Mean Wind Speed (knot)"].max().reset_index()
    daily_max_wind.rename(columns={"Mean Wind Speed (knot)": "Daily Max Wind Speed (knot)"}, inplace=True)
    
    # Calculate rolling average
    daily_max_wind["Rolling Avg Max Wind Speed (knot)"] = daily_max_wind["Daily Max Wind Speed (knot)"].rolling(window=30).mean()
    
    # Plot data
    axes[i].plot(daily_max_wind["Date"], daily_max_wind["Rolling Avg Max Wind Speed (knot)"], 
                 label=f"{year} - 30-Day Rolling Avg", color=colors[i])
    axes[i].set_title(f"{year}", fontsize=14)
    axes[i].set_ylabel("Wind Speed (knot)", fontsize=12)
    axes[i].grid(True, linestyle='--', alpha=0.7)
    axes[i].legend(fontsize=10)
    
# Common x-axis label
axes[-1].set_xlabel("Date", fontsize=12)
axes[-1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Convert degrees to radians for polar plot
df["Wind Direction Radians"] = np.deg2rad(df["Predominant Wind Direction (degree)"])

# Create polar plot
plt.figure(figsize=(10, 10)) 
ax = plt.subplot(111, polar=True)

# Use a colourmap
sc = ax.scatter(
    df["Wind Direction Radians"], 
    df["Mean Wind Speed (knot)"], 
    c=df["Mean Wind Speed (knot)"], 
    cmap="turbo", 
    alpha=0.7,                     
    s=40                            
)

# Customize the polar plot
ax.set_theta_zero_location("N")  # North at the top
ax.set_theta_direction(-1)       # Clockwise direction
ax.set_title("Polar Plot: Wind Speed vs Wind Direction", fontsize=16, weight='bold', pad=20)

# Add gridlines and label directions
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))  # 8 cardinal directions
ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.6) 
ax.set_yticks([10, 20, 30, 40])               
ax.set_yticklabels([f"{v} kt" for v in [10, 20, 30, 40]], fontsize=10)

# Add a color bar to represent wind speed
cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.1)
cbar.set_label("Mean Wind Speed (knot)", fontsize=12)

plt.show()




# thresholds
wind_speed_threshold = 45  # wind speed > 45 knots
precipitation_threshold = 15  # rain > 10 mm

# storms
storm_events = df[
    (df["Mean Wind Speed (knot)"] > wind_speed_threshold) |
    (df["Precipitation Amount (mm)"] > precipitation_threshold)
]

print(f"Number of potential storm events: {len(storm_events)}")


# count the storms
storm_days = storm_events.groupby("Date").size().reset_index(name="Storm Event Count")

# print
print(storm_days)


#Looking at how strong the storms are
storm_severity = storm_events.groupby("Date").agg({
    "Mean Wind Speed (knot)": "max",  # Max wind speed
    "Precipitation Amount (mm)": "sum"  # Total precipitation
}).reset_index()

storm_severity.rename(columns={
    "Mean Wind Speed (knot)": "Max Wind Speed (knot)",
    "Precipitation Amount (mm)": "Total Precipitation (mm)"
}, inplace=True)

print(storm_severity)



# Plot the data
ax = storm_severity.plot(
    x="Date", 
    y=["Max Wind Speed (knot)", "Total Precipitation (mm)"], 
    kind="bar",
    figsize=(14, 7),                
    color=["steelblue", "orange"],  
    width=0.8,                      
    legend=False                    
)

# Title and axis labels
ax.set_title("Storm Severity Metrics", fontsize=16, weight='bold', pad=20)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Value", fontsize=12)

# show every second x tick
ax.set_xticks(ax.get_xticks()[::2])  # 
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

# Add gridlines
ax.grid(axis="y", linestyle="--", alpha=0.7)  

# legend
ax.legend(["Max Wind Speed (knot)", "Total Precipitation (mm)"], 
          fontsize=10, loc="upper left", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()




# datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter data 
start_date = pd.to_datetime("2010-01-01") 
filtered_df = df[df['Date'] >= start_date]

# Select columns 
correlation_data = filtered_df[["Mean Wind Speed (knot)", "Air Temperature (C)", "Relative Humidity (%)"]]

# correlation matrix
correlation_matrix = correlation_data.corr()

# Create the heatmap
plt.figure(figsize=(10, 8)) 
sns.heatmap(
    correlation_matrix,
    annot=True,              
    fmt=".2f",                
    cmap="coolwarm",      
    linewidths=0.5,          
    annot_kws={"size": 12},    
    cbar_kws={"shrink": 0.8}  
)

#title and labels
plt.title("Correlation Matrix: Wind Speed, Temperature, Humidity (2010 Onwards)", fontsize=16, weight="bold", pad=20)
plt.xticks(fontsize=12, rotation=45, ha="right")  
plt.yticks(fontsize=12, rotation=0) 
plt.tight_layout()             
plt.show()




# make a season column
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

filtered_df["Season"] = filtered_df["Date and Time (UTC)"].dt.month.apply(get_season)

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df, x="Season", y="Mean Wind Speed (knot)", palette="Set2")
plt.title("Seasonal Distribution of Wind Speed")
plt.xlabel("Season")
plt.ylabel("Wind Speed (knot)")
plt.show()



#annual_mean_wind_speed = filtered_df.groupby('Year')['Mean Wind Speed (knot)'].mean().reset_index()


# trend line
future_years = np.arange(2025, 2035)  # 10 years into the future
future_trend = intercept + slope * future_years

# put it all together
extended_years = np.concatenate([annual_mean_wind_speed["Year"], future_years])
extended_trend = np.concatenate([
    intercept + slope * annual_mean_wind_speed["Year"], 
    future_trend
])

# Plot
plt.figure(figsize=(14, 7))
sns.lineplot(
    data=annual_mean_wind_speed, 
    x="Year", 
    y="Mean Wind Speed (knot)", 
    label="Annual Mean Wind Speed", 
    marker="o", 
    color="blue", 
    linewidth=2
)

# Plot the trend line with extrapolation
plt.plot(
    extended_years, 
    extended_trend, 
    color="red", 
    linestyle="--", 
    linewidth=2, 
    label="Trend Line (2005-2034)"
)

# Add confidence intervals (example assumes a constant margin, replace with actual CI if available)
ci_margin = std_err * 1.96  # 95% CI
lower_bound = extended_trend - ci_margin
upper_bound = extended_trend + ci_margin
plt.fill_between(extended_years, lower_bound, upper_bound, color="red", alpha=0.2, label="95% Confidence Interval")

# Add annotations for statistics
plt.annotate(
    f"Slope: {slope:.4f}\nP-value: {p_value:.4f}\nR²: {r_value**2:.4f}", 
    xy=(2006, min(annual_mean_wind_speed['Mean Wind Speed (knot)']) + 0.5), 
    fontsize=12, 
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
)

# Title and labels
plt.title("Trend in Annual Mean Wind Speeds with 10-Year Projection (2005-2034)", fontsize=18, fontweight="bold", pad=20)
plt.xlabel("Year", fontsize=14, labelpad=10)
plt.ylabel("Mean Wind Speed (knot)", fontsize=14, labelpad=10)

# Customize legend and grid
plt.legend(fontsize=12, frameon=True, loc="upper left", title="Legend", title_fontsize=12)
plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()



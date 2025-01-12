# PFDA-project

## Author: Aoife Flavin

# Wind Farm Location Analysis

This repository contains an analysis aimed at identifying the most suitable location for building a wind farm among three potential sites in Cork: Cork Airport, Moore Park, and Sherkin Island. The analysis is performed using weather data from these locations and leverages Python data analysis tools.

## Repository Contents

### CSV Files
- **`cork_airport_weather.csv`**: Weather data from Cork Airport.
- **`moore_park_weather.csv`**: Weather data from Moore Park.
- **`sherkin_island_weather.csv`**: Weather data from Sherkin Island.

### Jupyter Notebook
- **`wind.ipynb`**: A Jupyter Notebook that analyzes the weather data from the three locations to assess their suitability for a wind farm. This notebook includes:
  - Data loading and preprocessing.
  - Exploratory Data Analysis to understand wind patterns.
  - Statistical and machine learning techniques to evaluate the viability of each location.

### Requirements File
- **`requirements.txt`**: Contains the Python dependencies required to run the analysis. The listed packages are:
  - [`numpy`](https://numpy.org/)
  - [`pandas`](https://pandas.pydata.org/)
  - [`matplotlib`](https://matplotlib.org/)
  - [`seaborn`](https://seaborn.pydata.org/)
  - [`scikit-learn`](https://scikit-learn.org/stable/)
  - [`scipy`](https://scipy.org/)

## Getting Started

### Prerequisites
Ensure you have Python 3.8 or later installed on your system. Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Running the Analysis
1. Clone this repository to your local machine:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the repository folder:
   ```bash
   cd <repository_folder>
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook wind.ipynb
   ```
4. Run the notebook cells to perform the analysis.

## Features
- Comprehensive analysis of wind patterns across three locations.
- Visualisation of weather data using `matplotlib` and `seaborn`.
- Data preprocessing and statistical evaluation using `pandas`, `numpy`, and `scipy`.
- Machine learning insights using `scikit-learn` to assess wind farm viability.

## Future Improvements
- Incorporate additional data sources for more robust analysis.
- Extend the analysis to include more prediction models.



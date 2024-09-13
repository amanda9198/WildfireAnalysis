import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import folium
from folium.plugins import HeatMap

# Load the data
wfWeekly = pd.read_csv("1- cumulative-area-burnt-by-wildfires-by-week.csv")
wfAnnualy = pd.read_csv("2- annual-area-burnt-by-wildfires.csv")
area = pd.read_csv("3- share-of-the-total-land-area-burnt-by-wildfires-each-year.csv")
landArea = pd.read_csv("4- annual-area-burnt-per-wildfire.csv")
landCover = pd.read_csv("5- annual-burned-area-by-landcover.csv")

# Clean and prepare the data
wfWeekly.columns = wfWeekly.columns.str.strip()
wfWeekly['Year'] = pd.to_datetime(wfWeekly['Year'])
wfWeekly_long = pd.melt(wfWeekly, id_vars=['Entity', 'Code', 'Year'], 
                        var_name='YearBurnt', value_name='CumulativeAreaBurnt')
wfWeekly_long['YearBurnt'] = wfWeekly_long['YearBurnt'].str.extract('(\d{4})').astype(int)
afghanistan_data = wfWeekly_long[wfWeekly_long['Entity'] == 'Afghanistan']

# 1. Time Series Analysis
plt.figure(figsize=(15, 7))
for year in afghanistan_data['YearBurnt'].unique():
    year_data = afghanistan_data[afghanistan_data['YearBurnt'] == year]
    plt.plot(year_data['Year'], year_data['CumulativeAreaBurnt'], label=year)

plt.title('Cumulative Area Burnt by Wildfires in Afghanistan')
plt.xlabel('Date')
plt.ylabel('Cumulative Area Burnt (sq km)')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Yearly Totals and Trend Analysis
yearly_totals = afghanistan_data.groupby('YearBurnt')['CumulativeAreaBurnt'].max()
years = yearly_totals.index.astype(int)
slope, intercept, r_value, p_value, std_err = stats.linregress(years, yearly_totals)

plt.figure(figsize=(12, 6))
plt.bar(years, yearly_totals, alpha=0.7)
plt.plot(years, intercept + slope*years, color='red', label=f'Trend (R² = {r_value**2:.2f})')
plt.title('Total Area Burnt by Wildfires in Afghanistan by Year with Trend')
plt.xlabel('Year')
plt.ylabel('Total Area Burnt (sq km)')
plt.legend()
plt.show()

print(f"Trend: {slope:.2f} sq km/year, p-value: {p_value:.4f}")

# 3. Seasonal Analysis
if 'Date' in afghanistan_data.columns:
    afghanistan_data['Month'] = pd.to_datetime(afghanistan_data['Date']).dt.month
    monthly_totals = afghanistan_data.groupby('Month')['CumulativeAreaBurnt'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_totals.plot(kind='bar')
    plt.title('Total Area Burnt by Month in Afghanistan')
    plt.xlabel('Month')
    plt.ylabel('Total Area Burnt (sq km)')
    plt.show()

# 4. Land Cover Impact Analysis
if 'Land cover type' in landCover.columns:
    afghanistan_landcover = landCover[landCover['Entity'] == 'Afghanistan']
    total_burned_area = afghanistan_landcover['Burned area'].sum()
    afghanistan_landcover['Percentage'] = afghanistan_landcover['Burned area'] / total_burned_area * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Land cover type', y='Percentage', data=afghanistan_landcover)
    plt.title('Percentage of Burned Area by Land Cover Type in Afghanistan')
    plt.xlabel('Land Cover Type')
    plt.ylabel('Percentage of Total Burned Area')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# 5. Statistical Summary
print("Statistical Summary of Yearly Burnt Areas:")
print(yearly_totals.describe())

# 6. Box Plot of Yearly Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=yearly_totals)
plt.title('Distribution of Yearly Burnt Areas in Afghanistan')
plt.xlabel('Area Burnt (sq km)')
plt.show()

# 7. Correlation with Annual Data
annual_data = wfAnnualy[wfAnnualy['Entity'] == 'Afghanistan']
annual_data = annual_data.set_index('Year')
yearly_totals = yearly_totals.reindex(annual_data.index)

correlation = yearly_totals.corr(annual_data['Burned area'])
plt.figure(figsize=(10, 6))
plt.scatter(yearly_totals, annual_data['Burned area'])
plt.title(f'Correlation between Weekly and Annual Data (r = {correlation:.2f})')
plt.xlabel('Weekly Data: Total Area Burnt (sq km)')
plt.ylabel('Annual Data: Burned Area')
plt.show()

# 8. Heatmap of Wildfire Intensity (if geographical data is available)
# Note: This is a placeholder. You'll need actual latitude and longitude data for this to work.
if 'Latitude' in afghanistan_data.columns and 'Longitude' in afghanistan_data.columns:
    m = folium.Map(location=[33.93911, 67.709953], zoom_start=6)
    heat_data = [[row['Latitude'], row['Longitude'], row['CumulativeAreaBurnt']] for index, row in afghanistan_data.iterrows()]
    HeatMap(heat_data).add_to(m)
    m.save('afghanistan_wildfire_heatmap.html')
    print("Heatmap saved as 'afghanistan_wildfire_heatmap.html'")

# 9. Predictive Modeling (Simple Example)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = yearly_totals.index.values.reshape(-1, 1)
y = yearly_totals.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Prediction of Burned Area')
plt.xlabel('Year')
plt.ylabel('Burned Area (sq km)')
plt.show()

# 10. Comparative Analysis (if data for other countries is available)
# This is a placeholder. You'll need data for other countries to make this comparison.
countries_data = wfAnnualy[wfAnnualy['Entity'].isin(['Afghanistan', 'Pakistan', 'Iran'])]
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Burned area', hue='Entity', data=countries_data)
plt.title('Comparison of Burned Areas in Afghanistan and Neighboring Countries')
plt.xlabel('Year')
plt.ylabel('Burned Area (sq km)')
plt.show()

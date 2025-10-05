import pandas as pd

# Load the dataset
data = pd.read_csv("country.csv")
print(data.head())

# Sort countries by population in descending order
pop_descending = data.sort_values("Population", ascending=False)
print("Top 10 countries by population:")
print(pop_descending[["Country", "Population"]].head(10))

# Sort countries by GDP per capita in ascending order
gdp_ascending = data.sort_values("GDP ($ per capita)", ascending=True)
print("\nBottom 10 countries by GDP per capita:")
print(gdp_ascending[["Country", "GDP ($ per capita)"]].head(10))

# Countries with population over 10 million
pop_over_10m = data[data["Population"] > 10_000_000]
print(f"\nNumber of countries with population over 10 million: {len(pop_over_10m)}")
print("These countries are:")
print(pop_over_10m[["Country", "Population"]].sort_values("Population", ascending=False))

# Top countries by literacy rate
highest_literacy = data.sort_values("Literacy (%)", ascending=False)
print("\nTop 5 countries by literacy rate:")
print(highest_literacy[["Country", "Literacy (%)"]].head(5))

# Countries with GDP per capita above 10,000
gdp_over_10k = data[data["GDP ($ per capita)"] > 10_000]
print(f"\nNumber of countries with GDP per capita above $10,000: {len(gdp_over_10k)}")
print("These countries are:")
print(gdp_over_10k[["Country", "GDP ($ per capita)"]].sort_values("GDP ($ per capita)", ascending=False))

# Top countries by population density
highest_density = data.sort_values("Pop. Density (per sq. mi.)", ascending=False)
print("\nTop 10 countries by population density:")
print(highest_density[["Country", "Pop. Density (per sq. mi.)"]].head(10))

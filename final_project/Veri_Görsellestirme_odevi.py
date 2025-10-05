import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("50_Startups.csv")
print(data.head())

# R&D Spend vs Profit
plt.figure(figsize=(10, 6))
plt.scatter(data["R&D Spend"], data["Profit"], color="blue", alpha=0.7)
plt.xlabel("R&D Spend ($)")
plt.ylabel("Profit ($)")
plt.title("Relationship between R&D Spend and Profit")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Administration vs Profit
plt.figure(figsize=(10, 6))
plt.scatter(data["Administration"], data["Profit"], color="red", alpha=0.7)
plt.xlabel("Administration Spend ($)")
plt.ylabel("Profit ($)")
plt.title("Relationship between Administration Spend and Profit")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Average profit by state
avg_profit = data.groupby("State")["Profit"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(avg_profit.index, avg_profit.values, color=["skyblue", "lightgreen", "lightcoral"])
plt.xlabel("State")
plt.ylabel("Average Profit ($)")
plt.title("Average Profit by State")
plt.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f"${height:,.0f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# Distribution of spending types (boxplot)
spending_data = [data["R&D Spend"], data["Administration"], data["Marketing Spend"]]
labels = ["R&D Spend", "Administration Spend", "Marketing Spend"]

plt.figure(figsize=(12, 8))
box = plt.boxplot(spending_data, tick_labels=labels, patch_artist=True)

colors = ["lightblue", "lightgreen", "lightcoral"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.ylabel("Amount ($)")
plt.title("Distribution of Startup Spending Types")
plt.grid(axis="y", alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary statistics for spending types
print("Summary of Spending Types")
print("=" * 40)
for data_col, label in zip(spending_data, labels):
    print(f"{label}:")
    print(f"  Mean: ${data_col.mean():,.2f}")
    print(f"  Median: ${data_col.median():,.2f}")
    print(f"  Std Dev: ${data_col.std():,.2f}")
    print("-" * 25)


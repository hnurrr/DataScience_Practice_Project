# PSYCHOLOGY-THEMED DATA SCIENCE - WEEK 4
# NumPy and Pandas Applications for Psychology Research

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# SECTION 1: NUMPY ANALYSIS
# ========================
print("=== Brain Activity Matrix ===")
np.random.seed(42)
brain = np.random.randint(0, 101, (5, 5))
print(brain)
print(f"Mean: {brain.mean():.1f}, Std: {brain.std():.1f}, Max: {brain.max()}, Min: {brain.min()}, Diagonal sum: {np.trace(brain)}")

print("\n=== Psychology Test Scores Simulation ===")
np.random.seed(123)
scores = np.clip(np.random.normal(75, 15, 1000), 0, 100)
print(f"Mean: {scores.mean():.1f}, Median: {np.median(scores):.1f}, Std: {scores.std():.1f}")
print(f"Failing (<50): {(scores < 50).sum()} students")

# ========================
# SECTION 2: PANDAS ANALYSIS
# ========================
print("\n=== Psychology Student Database ===")
data = {
    'Student': ['Emma','Liam','Sophia','Noah','Olivia','William','Ava','James'],
    'Age':[20,21,19,22,20,23,19,21],
    'Specialization':['Clinical','Cognitive','Social','Clinical','Cognitive','Social','Clinical','Cognitive'],
    'Statistics':[85,78,92,88,76,82,90,85],
    'Research':[82,85,88,90,80,78,86,83],
    'Theory':[88,90,85,92,88,85,91,87],
    'Ethics':[90,87,89,85,92,88,89,86]
}
df = pd.DataFrame(data)

# Add GPA column
courses = ['Statistics','Research','Theory','Ethics']
df['GPA'] = df[courses].mean(axis=1)
print(df[['Student','Specialization','GPA']].round(1))

# Best specialization
specialization_avg = df.groupby('Specialization')['GPA'].mean()
best_spec = specialization_avg.idxmax()
print(f"Best specialization: {best_spec} ({specialization_avg.max():.1f} GPA)")

# High performers
high = df[df['GPA'] > 85]
print(f"High performers (>85 GPA): {list(high['Student'])}")

# ========================
# SECTION 3: VISUALIZATION
# ========================
plt.figure(figsize=(12,6))

# GPA by specialization
plt.subplot(1,2,1)
sns.barplot(x=specialization_avg.index, y=specialization_avg.values, palette="pastel")
plt.title("Average GPA by Specialization")
plt.ylabel("GPA")

# Student heatmap
plt.subplot(1,2,2)
sns.heatmap(df.set_index('Student')[courses], annot=True, cmap="YlOrRd", fmt='g')
plt.title("Student Scores Heatmap")

plt.tight_layout()
plt.show()

# ========================
# SUMMARY
# ========================
print("\n=== Summary ===")
course_means = df[courses].mean()
print(f"Overall class GPA: {df['GPA'].mean():.1f}")
print(f"Highest GPA: {df['GPA'].max():.1f}")
print(f"Lowest GPA: {df['GPA'].min():.1f}")
print(f"Most challenging course: {course_means.idxmin()} ({course_means.min():.1f})")
print(f"Easiest course: {course_means.idxmax()} ({course_means.max():.1f})")
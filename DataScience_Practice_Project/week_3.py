# PSYCHOLOGY-THEMED PYTHON HOMEWORK - WEEK 3
# Topics: Advanced Data Structures, Functions, Lambda, Modules

import random
import statistics

# Question 1 - Test Scores
scores = [85, 92, 76, 92, 100, 76, 85, 92]
print(f"Original: {scores}")
print(f"Unique: {list(set(scores))}, Max: {max(scores)}, Min: {min(scores)}, Sorted: {sorted(scores)}")

# Question 2 - Armstrong Number Checker
def is_armstrong(n):
    return n == sum(int(d)**len(str(n)) for d in str(n))

ids = [153, 371, 9474, 123, 407]
for id_num in ids:
    print(f"ID {id_num} {'IS' if is_armstrong(id_num) else 'IS NOT'} an Armstrong number")

# Question 3 - Skills Sets
clinical = {"SPSS", "R", "Statistics", "Counseling"}
research = {"Python", "SPSS", "Experimental_Design", "Statistics"}
print(f"Common skills: {clinical & research}")
print(f"Clinical only: {clinical - research}")
print(f"All skills: {sorted(clinical | research)}")

# Question 4 - Random Test Scores
rand_scores = [random.randint(1, 100) for _ in range(10)]
print(f"Random scores: {rand_scores}, Mean: {statistics.mean(rand_scores):.2f}, Std: {statistics.stdev(rand_scores):.2f}")

# Question 5 - Text Analyzer
def text_analyzer(text):
    words = text.lower().split()
    freq = {w: words.count(w) for w in words}
    return len(words), max(words, key=len), max(freq, key=freq.get), freq[max(freq, key=freq.get)]

text = "cognitive psychology studies mental processes including attention memory and perception"
total, longest, most_freq, count = text_analyzer(text)
print(f"Total: {total}, Longest: {longest}, Most frequent: '{most_freq}' (appears {count} times)")

# Question 6 - IQ Built-in Functions
iqs = [95, 112, 87, 118, 124, 93, 116]
even_iqs = list(filter(lambda x: x % 2 == 0, iqs))
squares = sorted(list(map(lambda x: x**2, even_iqs)), reverse=True)
print(f"Even IQs: {even_iqs}, Squares: {squares}")

# Question 7 - Lambda Sorting
terms = ["cognition", "behavior", "neuroscience", "psychotherapy", "emotion"]
print(f"Sorted by length: {sorted(terms, key=lambda w: len(w))}")

# Question 8 - Digit Sum Function
def digit_sum(s):
    return sum(int(c) for c in s if c.isdigit())

codes = ["EXP12A3B", "CTRL45C2", "abc123def456"]
for c in codes:
    print(f"{c} → digit sum: {digit_sum(c)}")

# MINI PROJECT - Research Paper Analysis
papers = [
    {"title":"CBT","author":"Smith","field":"Clinical","citations":1200,"year":2021},
    {"title":"Neural Networks","author":"Johnson","field":"Cognitive","citations":950,"year":2020},
    {"title":"Stats in Psychology","author":"Smith","field":"Methods","citations":700,"year":2019},
    {"title":"Deep Learning","author":"Williams","field":"Cognitive","citations":1800,"year":2022},
]

# Most cited paper
most_cited = max(papers, key=lambda p: p['citations'])
print(f"Most cited: '{most_cited['title']}' ({most_cited['citations']})")

# Author citations
authors = {}
for p in papers:
    authors[p['author']] = authors.get(p['author'], 0) + p['citations']
print(f"Author totals: {authors}")

# Fields and high-impact papers
fields = set(p['field'] for p in papers)
high_impact = [p['title'] for p in papers if p['citations'] > 1000]
print(f"Fields: {fields}, High-impact (>1000 citations): {high_impact}")

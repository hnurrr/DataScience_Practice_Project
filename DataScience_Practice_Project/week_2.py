# Topics: Conditionals, Loops, Strings, Lists

# ============================================================================
# QUESTIONS SECTION
# ============================================================================

# Question 1 - IQ Score Analysis
iq = int(input("\nEnter participant's IQ score: "))
sign = "Positive" if iq > 0 else "Negative" if iq < 0 else "Zero"
parity = "Even" if iq % 2 == 0 else "Odd"
print(f"IQ Score: {iq} → {sign}, {parity}")

print("\n" + "="*50) 

# Question 2 - Letter Frequency
term = input("\nEnter a psychology term: ").lower()
freq = {ch: term.count(ch) for ch in term if ch.isalpha()}
print(f"Letter frequency: {freq}")

print("\n" + "="*50) 

# Question 3 - Password Check
pwd = input("\nEnter lab password: ")
ok = len(pwd) >= 8 and any(c.isupper() for c in pwd) and any(c.isdigit() for c in pwd)
print("Password APPROVED " if ok else "Password REJECTED ")

print("\n" + "="*50)

# Question 4 - Reaction Time Analysis
times = [12, 4, 9, 25, 30, 7, 18]
avg = sum(times)/len(times)
above_avg = [t for t in times if t > avg]
print(f"Times: {times}, Avg: {avg:.2f}, Above Avg: {above_avg}")
 
print("\n" + "="*50)

# Question 5 - Stress Level Pattern
print("\nStress level visualization:")
for i in range(1, 6):
    print("*" * i)

print("\n" + "="*50)

# Question 6 - Anxiety Ratings
ratings = []
print("\nEnter anxiety ratings 1-10 (0 to finish):")
while (r := int(input())) != 0:
    ratings.append(r)
if ratings:
    print(f"Total ratings: {len(ratings)}, Sum: {sum(ratings)}, Average: {sum(ratings)/len(ratings):.2f}")
else:
    print("No ratings collected.")

print("\n" + "="*50)

# Question 7 - Palindrome Check
term2 = input("\nEnter a psychology term to check palindrome: ").lower().replace(" ", "")
print(f"'{term2}' {'IS' if term2==term2[::-1] else 'IS NOT'} a palindrome")

print("\n" + "="*50)

# Question 8 - Special Participant IDs
ids = [n**2 for n in range(1, 101) if n % 15 == 0]
print(f"\nSpecial IDs (squares of numbers divisible by 3 and 5): {ids}")

print("\n" + "="*50)

# Question 9 - Research Title Formatting
title = input("\nEnter research paper title: ")
formatted = " ".join(word.capitalize() for word in title.split())
print(f"Formatted title: {formatted}")

print("\n" + "="*50)

# MINI PROJECT: Therapy Feedback Analysis
feedbacks = [input(f"Feedback {i+1}: ") for i in range(int(input("\nHow many feedbacks? ")))]
lengths = [len(f) for f in feedbacks]
positive_words = ["good", "great", "excellent", "positive", "helpful", "better"]
positive_count = sum(any(pw in f.lower() for pw in positive_words) for f in feedbacks)

print("\n" + "="*50)

print("\nTHERAPY FEEDBACK RESULTS")
print(f"Total: {len(feedbacks)}, Positive: {positive_count}")
print(f"Longest: {max(feedbacks, key=len)}, Shortest: {min(feedbacks, key=len)}")
print(f"Average length: {sum(lengths)/len(lengths):.1f}")
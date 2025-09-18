# PSYCHOLOGY-THEMED PYTHON HOMEWORK: DATA TYPES AND OPERATORS

# ============================================================================
# SECTION 1: DATA TYPES
# ============================================================================

# Exercise 1: Psychology study participant information
print("=== Exercise 1: Participant Data ===")
name = input("Enter participant name: ")
age = int(input("Enter age: "))
height = float(input("Enter height (meters): "))

print(f"Participant: {name}, Age: {age}, Height: {height}m")
print(f"Data types: {type(name)}, {type(age)}, {type(height)}")

print("\n" + "="*50)

# Exercise 2: Psychology student grades
print("=== Exercise 2: Psychology Student Grades ===")
cognitive_psych = 85
social_psych = 92
research_methods = 78

average = (cognitive_psych + social_psych + research_methods) / 3.0
print(f"Grades: Cognitive={cognitive_psych}, Social={social_psych}, Research={research_methods}")
print(f"Average: {average:.2f} (type: {type(average)})")

print("\n" + "="*50)

# Exercise 3: Psychology term analysis
print("=== Exercise 3: String Analysis ===")
psychology_term = "Neuroplasticity"
print(f"Term: {psychology_term}")
print(f"First: {psychology_term[0]}, Last: {psychology_term[-1]}")
print(f"Length: {len(psychology_term)}, Reversed: {psychology_term[::-1]}")

print("\n" + "="*50)

# ============================================================================
# SECTION 2: OPERATORS
# ============================================================================

# Exercise 4: Mathematical Operations 
print("=== Exercise 4: Psychology Experiment Calculator ===")
participants = int(input("Enter number of participants: "))
average_score = float(input("Enter average test score: "))

total_score = participants * average_score

print(f"Total score (Multiplication): {total_score}")
print(f"Score if one participant leaves (Subtraction): {total_score - average_score}")
print(f"Score per participant (Division): {total_score / participants}")
print(f"Remainder if dividing total score by 10 (Modulus): {total_score % 10}")

print("\n" + "="*50)

# Exercise 5: Student pass/fail system
print("=== Exercise 5: Psychology Student Pass/Fail ===")
student_average = float(input("Enter student average: "))
result = "Passed" if student_average > 50 else "Failed"
print(f"Result: {result}")

print("\n" + "="*50)

# Exercise 6: License eligibility
print("=== Exercise 6: Psychology License Eligibility ===")
participant_age = int(input("Enter age: "))
eligibility = "Can get license" if participant_age >= 18 else "Cannot get license"
print(f"Status: {eligibility}")

print("\n" + "="*50)

# Exercise 7: Therapy session discount
print("=== Exercise 7: Therapy Session Pricing ===")
session_price = float(input("Enter session price: "))
discount_rate = float(input("Enter discount percentage: "))

discounted_price = session_price - (session_price * discount_rate / 100)
print(f"Original: ${session_price:.2f}")
print(f"Discounted: ${discounted_price:.2f}")

print("\n" + "="*50)

# Exercise 8: Age Eligibility with Logical Operators
print("=== Exercise: Age Eligibility (Logical Operators) ===")
age = int(input("Enter your age: "))

print(f"Age = {age}")
print(f"Eligible (18+): {age >= 18}")
print(f"Teenager (13-19): {age >= 13 and age <= 19}")
print(f"Child OR Teenager: {age < 13 or (age >= 13 and age <= 19)}")
print(f"Not Adult: {not (age >= 18)}")

# ============================================================================
# SECTION 3: MINI PROJECTS
# ============================================================================

# Exercise 9: Psychology book shopping cart
print("=== Exercise 9: Psychology Books Shopping ===")
book1_price = float(input("Enter price of 'Cognitive Psychology' book: "))
book2_price = float(input("Enter price of 'Social Psychology' book: "))
book3_price = float(input("Enter price of 'Research Methods' book: "))

total = book1_price + book2_price + book3_price
final_price = total * 0.9 if total > 200 else total

print(f"Total: ${total:.2f}")
if total > 200:
    print("10% discount applied!")
print(f"Final price: ${final_price:.2f}")

print("\n" + "="*50)

# Exercise 10: Age categories for psychological development
print("=== Exercise 10: Developmental Psychology Age Groups ===")
birth_year = int(input("Enter birth year: "))
age = 2024 - birth_year

print(f"Age: {age}, Category:", 
      "Child" if age <= 12 else 
      "Adolescent" if age <= 17 else 
      "Adult")
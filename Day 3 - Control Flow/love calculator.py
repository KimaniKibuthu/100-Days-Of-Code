'''
You are going to write a program that tests the compatibility between two people.

To work out the love score between two people:

Take both people's names and check for the number of times the letters in the word TRUE occurs. Then check for the number of times the letters in the word LOVE occurs. Then combine these numbers to make a 2 digit number.

For Love Scores less than 10 or greater than 90, the message should be:

"Your score is **x**, you go together like coke and mentos."

For Love Scores between 40 and 50, the message should be:

"Your score is **y**, you are alright together."

Otherwise, the message will just be their score. e.g.:

"Your score is **z**."

e.g.

name1 = "Angela Yu"

name2 = "Jack Bauer"

T occurs 0 times

R occurs 1 time

U occurs 2 times

E occurs 2 times

Total = 5

L occurs 1 time

O occurs 0 times

V occurs 0 times

E occurs 2 times

Total = 3

Love Score = 53

Print: "Your score is 53.

'''
# Welcome message
print('Welcome to Cupid\'s Haven')

# Define variables
check_against = 'true love'
your_name = (input('Your name: \n')).lower()
crush_name = (input('Crush\'s name: \n')).lower()

all_names = your_name + crush_name
list_names = [letter for letter in all_names]
true_count = (list_names.count('t')) + (list_names.count('r')) +(list_names.count('u')) + (list_names.count('e'))
love_count = (list_names.count('l')) + (list_names.count('o')) +(list_names.count('v')) + (list_names.count('e'))
combined_score = int((f'{true_count}') + (f'{love_count}'))

# Conditions
if combined_score < 10 or combined_score > 90:
    print(f'Your score is {combined_score}, you are like coke and menthos')

elif combined_score >=40 and combined_score <=50:
    print(f'Your score is {combined_score}, you are alright together')

else:
    print(f'Your score is {combined_score}')

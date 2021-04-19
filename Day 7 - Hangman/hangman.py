# Import the necessary modules 
import random
from hangman_art import logo, stages
from hangman_words import word_list

# Define variables
chosen_word = random.choice(word_list)
list_chosen_word = list(chosen_word)
length = len(list_chosen_word)
display = list(('_'*length))

# Start game
print(logo)

end_of_game = False
lives = 7
while not end_of_game:
    
    guess = (input('Guess a letter: \n')).lower()

    if guess in display:
        print('letter already used, choose another.')

    for index in range(len(list_chosen_word)):
        if list_chosen_word[index] == guess:
            display[index] = guess

    if guess not in list_chosen_word:
        print(stages[lives-1])
        lives-=1
        print('letter not in word')    
    
    print(display)


    if '_' not in display or lives == 0:
        end_of_game = True

        if '_' not in display:
            print('You win')
        elif lives == 0:
            print('You lose')
            print(f'The word is {chosen_word}')

    


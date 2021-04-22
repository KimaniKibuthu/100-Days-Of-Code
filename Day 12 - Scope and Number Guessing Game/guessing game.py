# Welcome to the game

import random
logo = ''' 
_______  __    __   _______     _______.     _______.   .___________. __    __   _______    .__   __.  __    __  .___  ___. .______    _______ .______      
 /  _____||  |  |  | |   ____|   /       |    /       |   |           ||  |  |  | |   ____|   |  \ |  | |  |  |  | |   \/   | |   _  \  |   ____||   _  \     
|  |  __  |  |  |  | |  |__     |   (----`   |   (----`   `---|  |----`|  |__|  | |  |__      |   \|  | |  |  |  | |  \  /  | |  |_)  | |  |__   |  |_)  |    
|  | |_ | |  |  |  | |   __|     \   \        \   \           |  |     |   __   | |   __|     |  . `  | |  |  |  | |  |\/|  | |   _  <  |   __|  |      /     
|  |__| | |  `--'  | |  |____.----)   |   .----)   |          |  |     |  |  |  | |  |____    |  |\   | |  `--'  | |  |  |  | |  |_)  | |  |____ |  |\  \----.
 \______|  \______/  |_______|_______/    |_______/           |__|     |__|  |__| |_______|   |__| \__|  \______/  |__|  |__| |______/  |_______|| _| `._____|
 
 '''

print(logo)
print('Welcome to the guessing game')
print('I am thinking of a number between 1 and 100. What could it be?')

number = random.randint(1,100)

choice = (input('Choose a dificulty: "hard" or "easy" \n')).lower()

if choice == 'hard':
    lives = 5
else:
    lives = 10


while lives > 0 :
    player_input = int(input('Guess a number:\n'))

    if lives > 0:
        if number > player_input :
            lives -= 1
            print(f'Too low, guess again. You have {lives} lives left')
            

        elif number < player_input:
            lives -= 1
            print(f'Too high, guess again. You have {lives} lives left')
        
        elif number == player_input:
            print('You guessed right')
            break

else:
    if number == player_input:
        print('You guessed right')
    else:
        print('You\'ve run out of lives. You lose') 


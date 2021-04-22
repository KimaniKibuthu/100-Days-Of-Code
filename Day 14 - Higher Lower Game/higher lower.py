# Necessary imports
from game_data import data
from art import vs, logo

# Welcome message
print('Welcome to the game')

# Define functions

# Taking comparisons
def user_response(vs, logo, data, index):
    name_a =data[index]['name']
    description_a = data[index]['description']
    country_a = data[index]['country']

    name_b =data[index+1]['name']
    description_b = data[index+1]['description']
    country_b = data[index+1]['country']
    print(logo)
    print(f'Compare A: {name_a}, a {description_a}, from {country_a}')
    print(vs)
    print(f'Agaist B: {name_b}, a {description_b}, from {country_b}')
    user_input = (input('Who has more followers: A or B \n')).lower()

    return user_input

def checker(user_input, data, index):
    a_followers = data[index]['follower_count']
    b_followers = data[index+1]['follower_count']


    if user_input == 'b':
        if b_followers > a_followers:
            return True
        else:
            return False
    else:
        if a_followers > b_followers:
            return True
        else:
            return False

def popper(checked, count, index, vs, logo, data):
    if checked == True:
        count +=1
        index +=2
        print(f'Your count is {count}')
        user_response(vs, logo, data, index)
        return index, count
    else:
        print(f'Sorry. You lose. Your total score is {count}')
        return index, count
        

def game(vs, logo, data):
    # Variables
    index = 0
    count = 0

    while (index + 2) < len(data):
        user_input = user_response(vs, logo, data, index)
        to_continue = checker(user_input, data, index)
        value, new_count = popper(to_continue, count, index, vs, logo, data)
        index += value
        count += new_count


        if to_continue == False:
            break
    else:
        print(f'You win, your count is {count}')  


game(vs, logo, data)
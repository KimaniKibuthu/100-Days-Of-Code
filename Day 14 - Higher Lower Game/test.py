print('Welcome')
from game_data import data
from art import logo, vs


for index in range(len(data)):
    while index + 2 < len(data):
        count = 0
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

        a_followers = data[index]['follower_count']
        b_followers = data[index+1]['follower_count']


        if user_input == 'b':
            if b_followers > a_followers:
                count+=1
                index+=1
            else:
                print(f'You lose. Your count is {count}')
                break
        else:
            if a_followers > b_followers:
                count +=1
                index+=1
            else:
                print(f'You lose. Your count is {count}')
                break

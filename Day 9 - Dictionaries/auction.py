# Welcome message
logo = '''
                         ___________
                         \         /
                          )_______(
                          |"""""""|_.-._,.---------.,_.-._
                          |       | | |               | | ''-.
                          |       |_| |_             _| |_..-'
                          |_______| '-' `'---------'` '-'
                          )"""""""(
                         /_________\\
                       .-------------.
                      /_______________\\
'''
print(logo)
print(' Welcome to bid!')

# define our clear function
def clear():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

# Auction
bids = {}
# Call for bids
other_bidders = True
while other_bidders == True:
    name = input('What is your name \n')
    bid = int(input('How much would you like to bid?\n $ '))

    # Store them in dictionary
    bids[name] = bid

    others = input('Are there any other bidders?\n ')
    if others == 'no':
        other_bidders = False
    else:
        other_bidders = True
        clear()
        
    
values = [value for value in bids.values()]
highest = max(values)

for key, value in bids.items():
    if highest == value:
        print(f'The highest bidder was {key} with a bid of $ {value}')


        


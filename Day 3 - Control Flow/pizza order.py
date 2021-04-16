#Congratulations, you've got a job at Python Pizza. Your first job is to build an automatic pizza order program.

#Based on a user's order, work out their final bill.

#Small Pizza: $15
#Medium Pizza: $20
#Large Pizza: $25
#Pepperoni for Small Pizza: +$2
#Pepperoni for Medium or Large Pizza: +$3
#Extra cheese for any size pizza: + $1

# Welcome message
print('Welcome to SS Pizza. I know you would like to make an order!!')

# The prices
large_bill = 25
medium_bill = 20
small_bill = 15

# Get input from customer
size = (input('What size of pizza would you like? (L, M or S) \n')).lower()
toppings = (input('Would you like any toppings with that? (yes or no)\n')).lower()

# Calculate bill
if toppings == 'no':
    if size == 'l':
        print(f'your total bill is {large_bill}')
    elif size == 'm':
        print(f'your total bill is {medium_bill}')
    else:
        print(f'your total bill is {small_bill}')

else:
    choice = (input('Would you like both cheese and pepperoni? (both or cheese or pepperoni )\n')).lower()

    if choice == 'both':
        if size == 'l':
            print(f'your total bill is {large_bill + 3 + 1}')
        elif size == 'm':
            print(f'your total bill is {medium_bill + 3 + 1}')
        else:
            print(f'your total bill is {small_bill + 2 + 1}')

    elif choice == 'cheese':
        if size == 'l':
            print(f'your total bill is {large_bill + 1}')
        elif size == 'm':
            print(f'your total bill is {medium_bill + 1}')
        else:
            print(f'your total bill is {small_bill + 1}')
    
    else:
        if size == 'l':
            print(f'your total bill is {large_bill + 3}')
        elif size == 'm':
            print(f'your total bill is {medium_bill + 3}')
        else:
            print(f'your total bill is {small_bill + 3}')







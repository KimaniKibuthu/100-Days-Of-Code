# Welcome message
print('Welcome to the Tip Calculator')

# Get the total bill
bill = float(input('How much was your bill?\n $ '))

# Get tip percentage
percentage = int(input('What percentage would you like to tip? (10, 12 or 15)\n'))

# People splitting the bill
people = int(input('How many people to split the bill?\n'))

# Calculate amount payable
amount_payable = round((bill * ((100 + percentage)/100)) / people, ndigits=2)

# Print it out
print(f'Each person will pay $ {amount_payable}')
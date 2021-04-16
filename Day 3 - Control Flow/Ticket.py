# Whether someone ride or not and whether they want a picture with the ride

# Welcome message 
print('Welcome to our roller coaster ride!!')

height = int(input('Kindly tell us your height in cm:\n'))

# Conditions
if height < 120:
    print('Bummer! You have to be more than 120 cm to ride. We will be glad to have you once you are taller')

else:
    print('Hooray! You can ride with us')

    age = int(input('Now tell us your age:\n'))
    answer = (input('Would you like a photo on your ride? (yes or no) \n')).lower()

    if age <= 12 and answer == 'yes':
        print('Your total price is $ 8')
    elif age <= 12 and answer == 'no':
        print('Your total price is $ 5') 

    elif age > 12 and age <=18 and answer == 'yes':
         print('Your total price is $ 10')
    elif age > 12 and age <=18 and answer == 'no':
         print('Your total price is $ 7')

    elif age > 18 and answer == 'yes':
         print('Your total price is $ 15')
    elif age > 18 and answer == 'no':
         print('Your total price is $ 12')


# BMI Calculator
print(' Welcome to our BMI Calculator')

height = float(input('What is your height in m?\n'))
weight = float(input('What is your weight in kg?\n'))

BMI = weight / (height ** 2)

if BMI <= 18.5:
    print('You are underweight')

elif BMI > 18.5 and BMI <= 25:
    print('You have a normal weight')

elif BMI > 25 and BMI <=30:
    print('You are overweight')

elif BMI > 30 and BMI <= 35:
    print('You are Obese')

elif BMI > 35:
    print('You are clinically obese')
# See whether the year is a leap year
# It is a leap year if it is divisible by 4 and not evenly divisible by 100 unless it is divisible by 400

# Input year
year = int(input('State your year: \n'))

# Check
if year % 4 == 0 and year % 100 != 0:
    print(f'The year {year} is a leap year')

elif year % 4 == 0 and year % 100 == 0:
    if year % 400 == 0:
        print(f'The year {year} is a leap year')
    else:
        print(f'The year {year} is not a leap year')

else:
    print(f'The year {year} is not a leap year')
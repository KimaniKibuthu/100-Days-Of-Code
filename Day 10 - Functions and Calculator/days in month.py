# Define Leap year
def is_leap(year):    
    if year % 4 == 0 and year % 100 != 0:
        return True

    elif year % 4 == 0 and year % 100 == 0:    
        if year % 400 == 0:
            return True
        else:
            return False

    else:
        return False


def days_in_month(year, month):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    months = ['january', 'february', 'march', 'april', 'may','june',
    'july','august', 'september', 'october', 'november', 'december'] 
    leap = is_leap(year)

    if leap == True and month == 'february':
        index = months.index(month)
        return month_days[index] + 1
        
    return month_days[months.index(month)]
     
print(days_in_month(year=int(input('Year:\n')), month=(input('Month:\n')).lower()))

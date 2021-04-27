from menu import MENU, resources


def prompter():
    return input('What would you like to have? (expresso/latte/cappuccino) \n')


def shut():
    return


def money_handler():
    print('Please insert coins')
    quarters = int(input('How many quarters? :'))
    dimes = int(input('How many dimes? :'))
    nickels = int(input('How many nickels? :'))
    pennies = int(input('How many pennies? :'))

    total = (0.25 * quarters) + (0.1 * dimes) + (0.05 * nickels) + (0.01 * pennies)
    return total


def report(money):
    # money = money_handler()
    for key, value in resources.items():
        print(f'{key}: {value}')
        print(f'money: $ {money}')


def resource_checker(choice):
    for key in MENU.keys():
        if key == choice:
            pick = MENU[key]['ingredients']

    for key, value in resources:
        while value >= pick[key]:
            return True
        else:
            print(f'Sorry, there isn\'t enough {key}')
            return False


def transaction(choice):
    money = money_handler()
    for key in MENU.keys():
        if key == choice:
            amount = MENU[key]['cost']

    if money < amount:
        amount_less = amount - money
        print(f'Sorry, less cash by ${amount_less}')
        return money_handler()

    elif money >= amount:
        change = money - amount
        print(f'Your change is ${change}')
        return True


def make_coffee():
    choice = prompter()
    resource = resource_checker(choice)
    to_make = transaction(choice)

    if not resource:
        print('Please refill')
    elif resource == True and to_make == True:
        print(f'Enjoy your {choice}')

    else:
        print('Can\'t make coffee')


make_coffee()

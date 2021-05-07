from menu import Menu
from coffee_maker import CoffeeMaker
from money_machine import MoneyMachine

# Instantiate objects
machine = CoffeeMaker()
costs = MoneyMachine()
menu_item = Menu()

is_on = True

while is_on:
    choice = input('What would you like? (espresso/latte/cappuccino)\n')

    if choice == 'report':
        machine.report()
        costs.report()

    elif choice == 'off':
        is_on = False

    else:
        drink = menu_item.find_drink(choice)
        if machine.is_resource_sufficient(drink):
            if costs.make_payment(drink.cost):
                machine.make_coffee(drink)
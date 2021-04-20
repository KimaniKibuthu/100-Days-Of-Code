# You are painting a wall. The instructions on the paint can says that 1 can of paint can cover 5 square meters of wall. Given a random height and width of wall, calculate how many cans of paint you'll need to buy.

#number of cans = (wall height ✖️ wall width) ÷ coverage per can.

#e.g. Height = 2, Width = 4, Coverage = 5

#number of cans = (2 ✖️ 4) ÷ 5
height = int(input('The height of the wall:\n'))
width = int(input('The width of the wall is: \n'))

def paint_counter(height, width):
    number_of_cans = (height * width) // 5
    if (height * width) % 5 != 0:
        number_of_cans += 1
    # Math.ceil rounds up a number
    print(f'You will need {number_of_cans} cans') 

paint_counter(height, width)

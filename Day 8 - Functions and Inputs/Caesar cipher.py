# Import logo
from art import logo

print(logo)

pre_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
alphabet = pre_alphabet + pre_alphabet

# Define the function

def caesar(text, shift, direction):
    text_list = list(text)
    new_list = []
    if direction == 'encode':
        for letter in text_list:
            if letter in alphabet:
                new_index = alphabet.index(letter) + shift
                new_list.append(alphabet[new_index])
            
            else:
                new_list.append(letter)
    
        encrypted_text = ''.join(new_list)
    
        return encrypted_text

    
    else:
        for letter in text_list:
            if letter in alphabet:
                new_index = alphabet.index(letter) - shift
                new_list.append(alphabet[new_index])
            
            else:
                new_list.append(letter)
    
        decrypted_text = ''.join(new_list)
    
        return decrypted_text

# Start the program
start = True
while start == True:
    direction = input("Type 'encode' to encrypt, type 'decode' to decrypt:\n")
    text = input("Type your message:\n").lower()
    shift = int(input("Type the shift number:\n"))

    while shift > 26:
        shift = int(input('shift should be less than 26. Input shift again: \n'))

    
    print(caesar(text, shift, direction))

    repeat = (input('Would you like to continue: (yes or no)\n')).lower()

    if repeat == 'yes':
        start = True
    else:
        start = False



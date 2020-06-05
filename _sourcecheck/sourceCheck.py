import socket
import os

def is_black_list(x):
    x_ip = socket.gethostbyname(x)
    

def Main():
    errors = []                       # The list where we will store results.
    linenum = 0
    substr = "from:".lower()          # Substring to search for.
    with open ('sample_email.txt', 'rt') as myfile:
        for line in myfile:
            linenum += 1
            if line.lower().find(substr) != -1:    # if case-insensitive match,
                errors.append(line.rstrip('\n'))

    for err in errors:
        x = err.split("@", 1)[1].split(" ", 1)
        if is_black_list(x):
            print("SPAM") 

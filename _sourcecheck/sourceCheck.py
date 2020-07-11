import socket
import os
import csv


def is_black_list(x):
    res = False
    try:
        x_ip = socket.gethostbyname(str(x[0]))
        print("IP: "+ x_ip)
        with open('full_blacklist_database.csv') as f:       #load the dataset
            reader = csv.reader(f)
            database1 = list(reader)
        database = str(database1)[1:-1] 
        if x_ip in database:
                res = True
        return res
    except Exception as e:
        print(e)
        return False
    

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
        else:
            print("HAM") 

if __name__ == '__main__':
    Main()
    
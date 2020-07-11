import os
import csv


def is_black_list(x):
    res = False
    with open('full_blacklist_database.csv') as f:       #load the dataset
        reader = csv.reader(f)
        database1 = list(reader)
    database = str(database1)[1:-1] 
    if x in database:
            res = True
    return res
    

def Main():
    res = True
    exitFlag = False
    with open ('sample_email.txt', 'r') as myfile:
        for line in myfile:
            if not exitFlag:
                for word in line.split():
                    if is_black_list(word):
                        res = False
                        exitFlag = True
                        break
                    else:
                        continue
            else:
                break
    if res:
        print("HAM")
    else:
        print("SPAM")
    

if __name__ == '__main__':
    Main()
    
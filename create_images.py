import PIL
import PIL.Image
import csv

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
counter = 0
prev = 0
with open("Data/Data.csv", 'r') as f:
    reader = csv.reader(f)
    for data in reader:
        alpha = int(data.pop(0))
        im = PIL.Image.new('L', (28, 28))
        im.putdata([int(i) for i in data])
        if alpha != prev:
            prev = alpha
            counter = 0
        im.save(f"Data/Letters/{ALPHABET[alpha]}/{ALPHABET[alpha]}_{counter}.png")
        counter += 1
    
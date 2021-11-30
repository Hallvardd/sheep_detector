from os import listdir
from shutil import copyfile

path_holtan = '/home/hallvard/sheep_finder/data/holtan/train/'
path_kari   = '/home/hallvard/sheep_finder/data/kari/train/'

path = path_holtan
#path = path_kari

labels = listdir(path + 'all_labels/')
images = listdir(path + 'images/')
counter = 0

for l in labels:
    for i in images:
        if l[:-4] == i[:-4]:
            copyfile(path + 'all_labels/' + l, path + 'labels/' + l )
            counter += 1

print(counter)
print(len(images))

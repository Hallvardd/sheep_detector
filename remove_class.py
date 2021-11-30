from os import listdir

src = 'holtan/train/labels_five_classes/'
dst = 'holtan/train/labels/'
filenames = listdir(src)

for fn in filenames:
    with open(src + fn, 'r') as f_in, open(dst + fn, 'w') as f_out:
        lines = [l for l in f_in if l[0] != 2]
        for i in range(len(lines)):
            if lines[i][0] == '4':
                lines[i] = '3' + lines[i][1:]
        f_out.write(''.join(lines))

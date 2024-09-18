import shutil
import os

folder = './519_3'
dest = './forlabel'
count = 0
for file in sorted(os.listdir(folder)):
    if count % 10 ==0:
        source = os.path.join(folder, file)
        destination = os.path.join(dest, file)
        shutil.copyfile(source, destination)
    count+=1 
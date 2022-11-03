import os
from PIL import Image

if __name__=="__main__":
    color_info = {}
    for root, dirs, files in os.walk("./data"):
        for name in files:
            if name.endswith('.jpg'):
                data = Image.open(os.path.join(root,name))
                color_info[data.mode] = color_info.get(data.mode, 0) + 1

    print(color_info)

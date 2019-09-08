import os

if __name__ == '__main__':
    fileDir = "./dogs-vs-cats/train/"
    pathDir = os.listdir(fileDir)
    print(pathDir[4][:3])
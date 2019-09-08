import os,shutil,random

def create_trainset():
    fileDir="./dogs-vs-cats/train/"
    dog_file = './DogsVSCats/train/dog/'
    cat_file = './DogsVSCats/train/cat/'
    pathDir=os.listdir(fileDir)
    #print(pathDir[1])
    for filename in pathDir:
        if filename[:3]=='dog':
            shutil.move(fileDir+filename,dog_file+filename)
        else:
            shutil.move(fileDir+filename,cat_file+filename)
    print('move OK!')
def create_validset():
    dog_file = './DogsVSCats/train/dog/'
    cat_file = './DogsVSCats/train/cat/'
    target_dog_file='./DogsVSCats/valid/dog/'
    target_cat_file='./DogsVSCats/valid/cat/'
    #分别从两类训练集中随机抽取2500张作为验证集
    pathdir_dog=os.listdir(dog_file)
    sample_dog=random.sample(pathdir_dog,2500)
    for name_dog in sample_dog:
        shutil.move(dog_file+name_dog,target_dog_file+name_dog)
    print('dog ok!')
    pathdir_cat=os.listdir(cat_file)
    sample_cat=random.sample(pathdir_cat,2500)
    for name_cat in sample_cat:
        shutil.move(cat_file+name_cat,target_cat_file+name_cat)
    print('cat ok!')
if __name__ == '__main__':
    create_validset()






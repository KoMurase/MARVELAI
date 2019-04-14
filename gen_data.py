#pillow 画像を扱うパイソンのパッケージ
from PIL import Image
import os,glob#glob:file一覧を扱うためのパッケージ
import numpy as np
from sklearn import model_selection
#交差検証：データを分離して学習と評価を行う手法

classes = ["CaptainAmerica","Ironman","SpiderMan"]
num_classes = len(classes)
image_size = 50

#画像の読み込み
X = []#画像の読み込み用
Y = []#ラベルデータ用
for index,classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i,file in enumerate(files):
        if i >= 200: break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y)
xy = (X_train,X_test,y_train,y_test)
np.save("./hero.npy",xy)#この画像データをモデル化する

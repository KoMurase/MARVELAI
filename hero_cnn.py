from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ["CaptainAmerica","Ironman","SpiderMan"]
num_classes = len(classes)
image_size = 50

#メイン関数を定義
def main():
    X_train,X_test,y_train,y_test = np.load("./hero.npy")
    #正規化
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train,num_classes)
    y_test = np_utils.to_categorical(y_test,num_classes)

    model = model_train(X_train,y_train)
    model_eval(model,X_test,y_test)

def model_train(X,y):
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=X.shape[1:]))
    #padding畳み込み結果が同じサイズになるように左右に足す
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))#クラスの数が３なので
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    #loss:損失関数（正解と推定値との誤差）

    model.fit(X,y,batch_size=32,epochs=50)#epoch行う回数遅いから少な目

    #モデルの保存
    model.save('./hero_cnn.h5')

    return model

def model_eval(model,X,y):
    scores = model.evaluate(X,y,verbose=1)
    print('Test loss',scores[0])
    print('Test Accuracy',scores[1])

if __name__=="__main__":
    main()

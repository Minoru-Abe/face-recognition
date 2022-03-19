from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np
import matplotlib.pyplot as plt

classes = ["tatsumi","reiri","tomomi"]
num_classes = len(classes)
image_size = 50

#メインの関数を定義する
def main():
    X_train, X_test, Y_train, Y_test = np.load("./kids.npy",allow_pickle=True)
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    model = model_train(X_train, Y_train, X_test, Y_test)
    model_eval(model, X_test, Y_test)

def model_train(X, Y, X_test, Y_test):
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=X.shape[1:]))
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
    # 下記DenseはInputの種類と合わせる必要がある。
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    #学習と、結果の記録
    #history = model.fit(X, Y, batch_size=32, epochs=10)
    history = model.fit(X, Y, validation_data=(X_test, Y_test), batch_size=32, epochs=500)

    #モデルの保存
    model.save('./kids_cnn.h5')

    #学習の可視化
    print(history.history)
    metrics = ['loss', 'acc']  # 使用する評価関数を指定
    plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意
    for i in range(len(metrics)):
        metric = metrics[i]
        plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
        plt.title(metric)  # グラフのタイトルを表示
        plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
        plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
        plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
        plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
        plt.legend()  # ラベルの表示
    plt.show()

    return model

def model_eval(model, X, Y):
    scores = model.evaluate(X, Y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

if __name__ == "__main__":
    main()

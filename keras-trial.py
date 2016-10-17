from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import numpy
seed = 9887
numpy.random.seed(seed)
import pandas as pd

train = pd.read_csv("data/train.csv")
train_lab = train['species']
train.drop(['id','species'],inplace=True,axis=1)
X_train, X_test, y_train, y_test = train_test_split(train, train_lab, test_size=0.4, random_state=0)


#test = pd.read_csv("data/test.csv")
#test_lab = test['species']
#test.drop(['id','species'],inplace=True,axis=1)


model = Sequential([
    Dense(32, input_dim=192),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=10, batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=16)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

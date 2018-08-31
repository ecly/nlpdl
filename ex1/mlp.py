from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(10)
dataset = numpy.loadtxt("diabetes.csv", delimiter=",", skiprows=1)

X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X, y, nb_epoch=150, batch_size=16)

predictions = model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)

scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

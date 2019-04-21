import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

# Generate dummy data
datagroup1 = (np.tile(np.arange(20),(1000,1))/19 + np.random.random((1000,20)))/2
datagroup2 = np.random.random((1000,20))
datagroup1_labeled = np.concatenate((datagroup1, np.ones((1000,1))), axis=1) 
datagroup2_labeled = np.concatenate((datagroup2, np.zeros((1000,1))), axis=1) 
alldata = np.concatenate((datagroup1_labeled, datagroup2_labeled), axis=0)
np.random.shuffle(alldata)
x_train = alldata[0:1500,0:20]
y_train = alldata[0:1500,20]
x_test = alldata[1500:2000,0:20]
y_test = alldata[1500:2000,20]
#print(x_train.shape)
#print(x_test.shape)

# Keras implementation
model = Sequential()
model.add(Dense(1, input_dim=20, activation='sigmoid'))

customizedadam = optimizers.Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=customizedadam, metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=100, batch_size=128)
accuracy_history = history.history['acc']
accuracy_history_last = accuracy_history[-1] * 100
print("Test Accuracy:", str("%.1f" % accuracy_history_last), "%")

score = model.evaluate(x_test, y_test, batch_size=64)
score_last = score[-1] * 100
print("Test Accuracy:", str("%.1f" % score_last), "%")
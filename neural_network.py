# Simple neural network for testing different input features for gesture perception
# Programmer: Paul Fletcher

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# DATA IMPORT FILES/DATAFRAMES
train_data = pd.read_csv('input_train.csv')
val_data = pd.read_csv('input_validate.csv')
test_data = pd.read_csv('input_test.csv')

# DROP UNNECESSARY COLUMNS
train_X = train_data.drop(columns=['list_pIDs', 'list_gestures', 'list_acc', 'list_conf', 'per_scores'], axis=1)
val_X = val_data.drop(columns=['list_pIDs', 'list_gestures', 'list_acc', 'list_conf', 'per_scores'], axis=1)
# test_X = test_data.drop(columns=['participant_ID', 'flight_date', 'mode_operation', 'state', 'control_mode', 'secs'], axis=1)
test_X = test_data.drop(columns=['list_pIDs', 'list_gestures', 'list_acc', 'list_conf', 'per_scores'], axis=1)

# PRINT DATAFRAMES
print(train_X.head())
print(test_X.head())

# ONE-HOT ENCODING OF THE CLASS DATA (I.E. SEMI-AUTO OR AUTO)
train_y = to_categorical(train_data.list_acc)
val_y = to_categorical(val_data.list_acc)
test_y = to_categorical(test_data.list_acc)

# CALCULATE THE APPROPRIATE WEIGHTED_LOSS BASED ON UNEVEN CLASS QUANTITIES IN DATA
y_integers = np.argmax(train_y, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))
print(d_class_weights)

# GET THE SHAPE OF THE DATAFRAME
n_cols = train_X.shape[1]
print('n_cols: ' + str(n_cols))

# CREATE THE MODEL
model = Sequential()

model.add(Dense(257, activation='relu', input_shape=(n_cols,)))
model.add(Dropout(0.2))
#kernel_regularizer=l2(0.01)
model.add(Dense(150, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# COMPILE THE MODEL
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# CREATE EARLY STOPPING MONITOR
early_stopping_monitor = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=15)

# FIT THE MODEL AND COLLECT THE HISTORY
history1 = model.fit(train_X, train_y, epochs=10, validation_data=(val_X, val_y), class_weight=d_class_weights)


with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# EVALUATE MODEL PERFORMANCE ON TEST_DATA
test_loss, test_acc = model.evaluate(test_X, test_y)
print('\nTest Loss: ' + str(test_loss) + ' Test Accuracy: ' + str(test_acc))
print('One Network Finished')

# PRINT ACCURACY AND LOSS GRAPHS
figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(history1.history['acc'], color='r')
plt.plot(history1.history['val_acc'], color='r', linestyle='dashed')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(history1.history['loss'], color='g')
plt.plot(history1.history['val_loss'], color='g', linestyle='dashed')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

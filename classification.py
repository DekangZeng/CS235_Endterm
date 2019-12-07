import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers






def Max_Min_Norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def spilt_labels(dataset):
    label=dataset[ :,0]

    label=label.reshape(-1,1)
    data = dataset[ :,1:]
    data = data.reshape(-1, 28, 28, 1)
    return data,label

def spilt_to_five(dataset):
    length = int(dataset.shape[0] * 0.2)
    A = dataset[0:length,:]
    B = dataset[length:length * 2,:]
    C = dataset[length * 2:length * 3,: ]
    D = dataset[length * 3:length * 4,: ]
    E = dataset[length * 4:, ]
    return A, B, C, D, E

def train_model():
    model=keras.Sequential([
        layers.Conv2D(16,3,padding='same',activation='relu',input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10,activation='softmax')
    ])
    optimizer=tf.keras.optimizers.SGD(0.001)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model
def K_fold():
    train=pd.read_csv('train.csv')
    train_data=np.array(train)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    data_A, data_B, data_C, data_D, data_E =spilt_to_five(train_data)
    data_list=[data_A, data_B, data_C, data_D, data_E]
    accuracy_list=[]
    loss_list=[]
    index=0
    for testset in data_list:
        train_data_list = data_list.copy()
        del train_data_list[index]
        trainset = np.vstack((train_data_list[0], train_data_list[1], train_data_list[2], train_data_list[3]))
        index += 1

        test_example, test_label = spilt_labels(testset)
        train_example, train_label = spilt_labels(trainset)
        train_example=Max_Min_Norm(train_example)
        test_example = Max_Min_Norm(test_example)
        model =train_model()
        model.summary()
        model.fit(x=train_example, y=train_label, epochs=200,
              validation_data=[test_example, test_label],
                  callbacks=[early_stop])
        loss,accuracy= model.evaluate(test_example, test_label)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
    return loss_list,accuracy_list

loss,accuracy=K_fold()
print(loss,accuracy)
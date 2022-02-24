# -*- coding: utf-8 -*-
"""
Project: 基於深度學習的AOI影像分類
Author: David Li
Create date:2022.02.17
Module function: 自行設定的簡化版CNN網路，共9層
"""
import tensorflow as tf
import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import time 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import KFold


#-----------------上傳訓練過程到TensorBoard(網路上)---------------------------------

NAME= "AOI_defect_CNN_6-{}".format(int(time.time()))
tensorboard  = TensorBoard(log_dir="logs/{}".format(NAME),histogram_freq=1)
"""
class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
        """
#-----------------訓練過程基本參數設置

#未補齊dataset前的檔案路徑
#trainpath="D:\\AOI_Defect_DataSet\\train_images\\train_images\\train"
#validpath="D:\\AOI_Defect_DataSet\\train_images\\train_images\\valid"

#補齊後的檔案路徑
trainpath="D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\Balanced_Dataset\\Train"
validpath="D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\Balanced_Dataset\\Valid"

savepath="D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\Model\\Model_Weight"
checkpointpath = "D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\Model\\Check_Point\\"

epoch=5
batchsize = 6
resize = (512,512)

#---------------讀入training data & validation data--------------------------------------------------
train=ImageDataGenerator(rescale=1/255)
validation =ImageDataGenerator(rescale= 1/255)
train_dataset = train.flow_from_directory(trainpath,
                                          target_size = resize,
                                          batch_size = batchsize,
                                          class_mode="categorical")


validation_dataset= train.flow_from_directory(validpath,
                                              target_size = resize,
                                              batch_size = batchsize,
                                              class_mode="categorical")

print(train_dataset.classes)



model = tf.keras.models.Sequential([
                                    #第一層一定要指定input形狀
                                    #tf.keras.layers.Dense(16, activation="relu", input_shape=(512,512,3)),
                                    
                                    tf.keras.layers.Conv2D(16, (3, 3),activation = "relu", input_shape=(512,512,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                      
                                    tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    
                                    tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    #tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Flatten(),
                                    
                                    tf.keras.layers.Dense(1024, activation="relu"),
                                    tf.keras.layers.Dropout(0.5),
                                    #tf.keras.layers.Dense(64, activation="relu"),
                                    tf.keras.layers.Dense(1024, activation="relu"),
                                    #tf.keras.layers.Dense(32, activation="relu"),
                                    
                                    tf.keras.layers.Dense(6,activation="softmax"),
                                    
                                      ]
                                     )




model.compile(loss="categorical_crossentropy",#binary_crossentropy
              #'sparse_categorical_crossentropy'
              optimizer ="adam",#adam
              metrics =["accuracy"]
              )

#-------------------回乎函式(callback)---------------------------------------------


#提早停止訓練
my_callbacks = [
    EarlyStopping(patience=5, monitor="val_accuracy", mode="auto")
    ]

#每10個epoch設一次檢查點並儲存最佳的權重
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpointpath, 
                                                                save_weights_only=True,
                                                                save_best_only=True)]

#------------------model執行------------------------------------------------------------
# --------------載入最近的檢查點的權重，接續上次繼續訓練------------------------------
model.load_weights(checkpointpath)


history = model.fit(train_dataset,
                    epochs=epoch,
                    validation_data=validation_dataset,
                    callbacks = model_checkpoint_callback,  #my_callbacks
                    #callbacks = [tensorboard]#,LRTensorBoard
                    verbose = 1
                    #verbose:訓練進度條訊息
                            #0:不輸出進度訊息
                            #1:顯示進度條
                            #2:每個epoch輸出一行紀錄
                    
                    )



model.summary()

#--------------------印出ACC和LOSS的圖---------------------------
print(history.history.keys())
    # summarize history for accuracy
    

ep=len(history.history['accuracy'])
vep=len(history.history['val_accuracy'])
ptac=[]
ptvl=[]
eptick=[]

for i in range(ep):
    ptac.append(history.history['accuracy'][i]*100)

for j in range(vep):
    ptvl.append(history.history['val_accuracy'][j]*100)

for k in range(epoch):
    if k%5==0:
        eptick.append(k)

plt.plot(ptac)
plt.plot(ptvl)
plt.title('model accuracy')
plt.ylabel('Accuracy(%)')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.xticks(eptick)
plt.xlim(0,epoch)
plt.ylim(0,100)
plt.grid(True)
plt.grid(color="black",linestyle=":")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.xticks(eptick)
plt.xlim(0,epoch)
plt.grid(True)
plt.grid(color="black",linestyle=":")
plt.show()

model.save(str(savepath)+'\\CNN_Model_0223.h5')

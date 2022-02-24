# -*- coding: utf-8 -*-
"""
Project: 基於深度學習的AOI影像分類
Author: David Li
Create date:2022.02.17
Module function: 丟入測試照片，預測照片種類，並輸出CSV檔
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2
import os 
import csv
import shutil

modelpath = "D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\\Model\\Model_Weight\\CNN_Model_0223.h5"
testpath = "D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\AOI_dataset\\test_images\\test_images"
littleset = "D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\AOI_dataset\\little_set"
resultpath = "D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\Model\\Tesdting_Result\\"
destpath = "D:\\1_NSYSU\\1_MasterDegree\\[Paper]\\AOI_dataset\\5"

Categories = ["Normal","Void","H-Line","V-Line","Edge","Particle"]

setlist = os.listdir(testpath)

model =load_model(modelpath, compile = True)
model.summary()

def prepare(testpath):
    IMG_SIZE = 512
    img_array = cv2.imread(testpath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)



def predict(testpath, setlist):
    prelist = []
    for i in range(len(setlist)): 
        imgs = cv2.imread(testpath+"\\"+ setlist[i])
        imgs = imgs/255
        X = np.expand_dims(imgs, axis=0)
        img = np.vstack([X])
        prediction = model.predict(img)
        classes = np.argmax(prediction)
        print("_______")
        print(classes)
        #defect = Categories[classes]
        prelist.append(classes)

            
    return prelist
        
        
  
ResultList = predict(testpath, setlist)

"""
#用於分類test出來的檔案
for i in ResultList:
    shutil.copy(testpath+"\\"+i, destpath+"\\"+i)
    print(i)
print(ResultList)
"""


if __name__ == "__main__":
    with open (resultpath + "test20220223.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = ["ID","Label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for j in range(len(setlist)):
            writer.writerow({"ID":setlist[j], "Label":ResultList[j]})







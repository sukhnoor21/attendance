# importing the necessary libraries
import cv2
import os
import numpy as np
import face_recognition 
from datetime import datetime, date


# Importing the path of images
path = r"C:\Users\hp\Desktop\pr\Attendance\test"
#This list will contain all images from the dataset
images = [] 
#this list will contain all the names from dataset
classNames = []
# used to get the list of all files and directories in the specified directory
#basically ye saari images mylist naam ke variable mei daal raha hai
myList = os.listdir(path)
#phir hum woh saari images ka naam print kr rhe hai
#also we are printing the extension
print(myList)
#basically we are applying for loop to split the full name of the images into root and extension
for cl in myList:
  curImg = cv2.imread(f'{path}/{cl}')#reading the name of all the images
  #appending the images list with the names that we read from the above line ie the curImg ....
  # This takes place along with the extension
  images.append(curImg)
  # splittext here splits the name of the image into root and extension
  #and then appending only root in a list that we made before called classNames
  classNames.append(os.path.splitext(cl)[0])
print(classNames)
#height, width = image.shape[:2]
  
# display width and height
#print("The height of the image is: ", height)
#print("The width of the image is: ", width)

# this function performs encoding of all the images in the dataset
def findEncodings(imgs):
  encodeList = []
  for img in imgs:
    # the predefined colour scheme in python is bgr...we are converting it to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #saari jo orientations hain unko theek krne ke liye
    encode = face_recognition.face_encodings(img)[0]
    #appending the encoded images to a list called encode list
    encodeList.append(encode)
  return encodeList
# opening the csv file to mark attendance
def markAttendance(name):
  x = date.today()
  with open('attendance_{}-{}-{}.csv'.format(x.day,x.month,x.year), 'a+') as f:#file open/append kr raha hai as f
    f.seek(0)
    dataList = f.readlines()#coverting into list
    nameList = []
    for line in dataList: 
      entry = line.split(',')
      nameList.append(entry[0])
    if name not in nameList:#the names that are already there will be not appended....if not, then appended
      now = datetime.now()
      dt = now.strftime('%H:%M:%S')
      d = date.today()
      f.writelines(f'\n{name},{dt},{d}')#appending the name, time and date in tne csv file


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))#printing the total number of encoded images in our dataset


capture = cv2.VideoCapture(0)

while True:
  success, img = capture.read()
  imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
  imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)#resizing and changing the colour scheme of the img from the webcam

  faceLocCur = face_recognition.face_locations(imgS)
  encodeCur = face_recognition.face_encodings(imgS, faceLocCur)

  for encodeFace, faceLoc in zip(encodeCur, faceLocCur):
    faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
    
    matchIndex = np.argmin(faceDistance)

    if faceDistance[matchIndex]:
      name = classNames[matchIndex].upper() 

      y1,x2,y2,x1 = faceLoc
      y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
      cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
      cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
      cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
      markAttendance(name)
  cv2.imshow('Webcam', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


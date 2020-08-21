import tkinter as tk
from threading import Thread, Event
import numpy as np
import imutils
import cv2
import tensorflow as tf
from PIL import Image
from PIL import ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# import winsound

import datetime
import time
import os

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
a = 0
names = ['None', 'zeynoc']

print("[INFO] loading face detector model...")
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model('mask_detector_v1.model')

cam.set(3, 1000)
cam.set(4, 850)
recognizer.read('trainer.yml')
print("\n [INFO] Recognizer readed")


class MASK(tk.Frame):

    # init function
    def __init__(self, parent):
        tk.Frame.__init__(self)
        self.parent = parent
        self.parent.geometry('1200x720+10+10')
        self.parent.wm_title("Mask Detector")

        self.image_tk0 = ImageTk.PhotoImage(Image.open("startup.png"))

        self.startwindow = tk.Label(self)
        self.startwindow.pack()
        self.startwindow.configure(image=self.image_tk0)

        # Define a quit button and quit event to help gracefully shut down threads
        tk.Button(self, text="ADD", command=self.add_persons).pack(side=tk.LEFT)

        tk.Button(self, text="MASK", command=self.mask_detection).pack(side=tk.LEFT)
        # tk.Button(self,text="Quit_mask",command=self.quit_mask).pack(side=tk.LEFT)

        tk.Button(self, text="SHOW", command=self.show_frame).pack(side=tk.LEFT)
        # tk.Button(self,text="Quit_show",command=self.quit_show).pack(side=tk.LEFT)

        tk.Button(self, text="Exit", command=self.parent.destroy).pack(side=tk.RIGHT)

        # self._quit_show = Event()
        # self._quit_mask = Event()

        self.capture_thread1 = None
        self.capture_thread2 = None
        self.capture_thread3 = None

    # This function launches a thread to do video capture
    def mask_detection(self):
        # self._quit_mask.clear()
        self.vidfeed1 = tk.Toplevel(self)
        self.vidfeed1.wm_title("MASK DETECTION FEED")

        self.lmain1 = tk.Label(self.vidfeed1)
        self.lmain1.grid(row=0, column=0)

        # Create and launch a thread that will run the video_capture function
        self.capture_thread1 = Thread(target=mask_detect, args=(self.vidfeed1, self.lmain1))
        self.capture_thread1.daemon = True
        self.capture_thread1.start()

    def add_persons(self):
        self.vidfeed2 = tk.Toplevel(self)
        self.vidfeed2.geometry('1200x720+10+10')
        self.vidfeed2.wm_title("ADD persons")

        self.lmain2 = tk.Label(self.vidfeed2)
        self.lmain2.pack(side=tk.TOP)
        self.image_tk1 = ImageTk.PhotoImage(Image.open("vid.png"))
        self.lmain2.configure(image=self.image_tk1)

        self.tb = tk.Text(self.vidfeed2)
        self.tb.pack(side=tk.LEFT)

        self.face_id = tk.IntVar(self.vidfeed2)

        self.e = tk.Entry(self.vidfeed2)
        self.e.pack(side=tk.LEFT)
        self.e.focus_set()

        # button to get id num of the person
        self.ok_btn = tk.Button(self.vidfeed2, text="OK", command=lambda: self.face_id.set(self.e.get()))
        self.ok_btn.pack(side=tk.LEFT)

        # Create and launch a thread that will run the video_capture function
        self.capture_thread2 = Thread(target=add, args=(self.e, self.vidfeed2, self.lmain2
                                                        , self.tb, self.ok_btn, self.face_id))
        self.capture_thread2.daemon = True
        self.capture_thread2.start()

    def show_frame(self):
        # self._quit_show.clear()
        self.vidfeed = tk.Toplevel(self)
        self.vidfeed.wm_title("CAMERA FEED")

        self.lmain = tk.Label(self.vidfeed)
        self.lmain.grid(row=0, column=0)

        # Create and launch a thread that will run the video_capture function
        self.capture_thread3 = Thread(target=show, args=(self.vidfeed, self.lmain))
        self.capture_thread3.daemon = True
        self.capture_thread3.start()

    """
    def quit_show(self):
        self._quit_show.set()
        try:
            self.lmain.destroy()
            self.vidfeed.destroy()

        except TypeError:
            pass


    def quit_mask(self):
        self._quit_mask.set()
        try:
            self.lmain1.destroy()
            self.vidfeed1.destroy()

        except TypeError:
            pass

    """


########  Helper functions  #######
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            '''
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                detect_and_predict_mask(frame, faceNet, maskNet)
            '''
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        preds = maskNet.predict(faces)
        tf.keras.backend.clear_session()
    return (locs, preds)


####### Key functions from buttons  ########

def add(e, vidfeed2, lmain2, tb, ok_btn, face_id):
    # For each person, enter one numeric face id
    tb.insert(tk.END, "For each person, enter one numeric face id \nin the box and press OK")
    # face_id = input('\n enter user id end press <return> ==>  ')
    print("face_id")
    ok_btn.wait_variable(face_id)

    print(face_id.get())

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # tb.delete(1.0, tk.END)
    tb.insert(tk.END, "\n$: Initializing face capture. Look the camera and wait ...")

    i = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id.get()) + '.' + str(i) + ".jpg",
                        gray[y:y + h, x:x + w])
            cv2image = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)

            imgtk = ImageTk.PhotoImage(image=img)
            lmain2.imgtk = imgtk
            lmain2.configure(image=imgtk)

            i = i + 1
            print(i)

        if i == 30:
            break

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    # tb.delete(1.0, tk.END)
    tb.insert(tk.END, "\n$: Training faces. It will take a few seconds. Wait ...")

    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.save('trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    # tb.delete(1.0, tk.END)
    tb.insert(tk.END, "\n$: faces trained. Exiting Program")

    time.sleep(2)


def show(vidfeed, lmain):
    (grabbed, frame) = cam.read()
    frame = imutils.resize(frame, width=900)
    label2 = "{} ".format("Detection Status : OFF ")
    label3 = "{} ".format(datetime.datetime.now())
    cv2.putText(frame, label2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(frame, label3, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    # cv2.imshow("Frame2", frame)
    # key = cv2.waitKey(1) & 0xFF
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    vidfeed.after(10, show(vidfeed, lmain))


# This function simply loops over and over, printing the contents of the array to screen
def mask_detect(vidfeed1, lmain1):
    tf.keras.backend.clear_session()
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    k = 0
    c1 = 0
    c2 = 0
    id = 0

    (grabbed, frame) = cam.read()
    frame = imutils.resize(frame, width=900)

    try:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        tf.keras.backend.clear_session()
    except cv2.error as e:
        mask_detect(vidfeed1, lmain1)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        if mask > withoutMask:
            k = 1
            c1 = c1 + 1
        else:
            c2 = c2 + 1
        if c2 > 0:
            if c1 == 0:
                label = "No Mask"
                if max(mask, withoutMask) * 100 > 90:
                    k = k + 1
                    # winsound.Beep(1000, 30)
                    c2 = 0
                    # ********* face detect********************************************************
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(int(minW), int(minH)),
                    )
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        if (confidence < 100):
                            id = names[id]
                        else:
                            id = "unknown"
                            confidence = "  {0}%".format(round(100 - confidence))
                        label1 = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                        label2 = "{}: {:.2f}%".format(str(id), round(100 - confidence))
                        cv2.putText(frame, label1, (x + 5, y - 5), font, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, label2, (x + 5, y + h - 5), font, 0.7, (0, 0, 255), 2)
                        label1 = "{}: {}: {:.2f}%".format("Mask Status", label, max(mask, withoutMask) * 100)
                        label2 = "{}: {}: {:.2f}%".format("Person", id, round(100 - confidence))
                        label3 = "{} ".format(datetime.datetime.now())
                        label4 = "{} ".format("Detection Status : ON ")
                        cv2.putText(frame, label4, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                        cv2.putText(frame, label1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
                        cv2.putText(frame, label2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 0, 0), 2)
                        cv2.putText(frame, label3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if c1 != 0:
            label = "Mask"
            color = (0, 255, 0)
            label1 = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            label4 = "{} ".format("Detection Status : ON ")
            label3 = "{} ".format(datetime.datetime.now())
            cv2.putText(frame, label4, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(frame, label1, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.putText(frame, label1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, label3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            c1 = 0

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img)
    lmain1.imgtk = imgtk
    lmain1.configure(image=imgtk)
    vidfeed1.after(10, mask_detect(vidfeed1, lmain1))


if __name__ == "__main__":
    root = tk.Tk()

    selectors = MASK(root)

    selectors.pack()

    root.mainloop()

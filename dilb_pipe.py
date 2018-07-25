# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:48:54 2018

@author: ANIRBAN
"""

import cv2
import numpy as np
import os
import glob


class Detect():
    CASCADE = "Face_cascade.xml"
    FACE_CASCADE = cv2.CascadeClassifier(CASCADE)
    i = 0
    # output_folder = ""
    # complete_path = glob.glob('Network/train_data/amit_original/*.jpg')
    # total_images =
    global_face = []
    unique_vector = []

    def detect_faces(this, image_new):
        image = image_new
        # print(1)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = this.FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(75, 75),
                                                   flags=0)
        # if not os.path.isdir(output_folder):
        # os.makedirs(output_folder)

        for x, y, w, h in faces:
            this.i = this.i + 1
            sub_img = image[y - 7:y + h + 2, x - 7:x + w + 7]
            # os.chdir(output_folder)

            try:
                sub_img = cv2.resize(sub_img, (224, 224))
                this.global_face = sub_img

                # cv2.imwrite(str(this.i) + ".jpg", sub_img)
            except:
                print("error")
            finally:
                pass
                # os.chdir("../")
                # this.global_face = np.asarray(this.global_face)
                # return this.global_face
                # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # cv2.imshow("Faces Found", image)
        # if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
        #     cv2.destroyAllWindows()
        return this.global_face

    def __init__(self, input_images):
        self.complete_images = input_images
        # self.complete_path = glob.glob(self.complete_path)

        self.no_of_images = input_images.shape[0]
        # print(self.no_of_images)
        for counter in range(0, self.no_of_images):
            # print(self.complete_images[counter])
            one = self.detect_faces(self.complete_images[counter])
            self.unique_vector.append(one)

        self.unique_vector = np.asarray(self.unique_vector)



        # obj = Detect(input_images, "non_amit_train1")
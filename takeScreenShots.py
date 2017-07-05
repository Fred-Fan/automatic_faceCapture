from PIL import ImageGrab
import time
import numpy as np
import imutils
import cv2

#clear the folder to avoid recapturing  images
import os, shutil
#Either put your folder address in this line or comment this part and delete manually
folder = '/Users/kaiyuewang/Downloads/face-alignment/images'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


i = 0
while(True):
    img = ImageGrab.grab() #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion
    # img_np = imutils.resize(img_np, width=800)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    path = "images/screenShot_" + str(i) + ".jpg"
    cv2.imwrite(path, frame)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    i += 1
    time.sleep(5)

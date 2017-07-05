# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

iter = 0
imageIter = 0

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--prefix-name", required=True,
	help="prefix of the current batch")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
while (True):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	fa = FaceAligner(predictor, desiredFaceWidth=120)

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread("images/screenShot_" + str(imageIter) + ".jpg")
	# image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# show the original input image and detect faces in the grayscale
	# image
	cv2.imshow("Input", image)
	rects = detector(gray, 2)

	# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=120)
		faceAligned = fa.align(image, gray, rect)



		# display the output images
		# cv2.imshow("Original", faceOrig)
		cv2.imwrite("inputData/" + args["prefix-name"] + str(iter) + ".jpg", faceAligned)
		cv2.imshow("Aligned", faceAligned)
		iter += 1

	imageIter += 1
	print("You are processing image No: {}".format(imageIter))

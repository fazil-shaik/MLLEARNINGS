# import cv2 as cv
# import numpy as np

# cap = cv.VideoCapture(0)

# while(1):

#     # Take each frame
#     _, frame = cap.read()

#     # Convert BGR to HSV
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

#     # define range of blue color in HSV
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])

#     # Threshold the HSV image to get only blue colors
#     mask = cv.inRange(hsv, lower_blue, upper_blue)

#     # Bitwise-AND mask and original image
#     res = cv.bitwise_and(frame,frame, mask= mask)

#     cv.imshow('frame',frame)
#     cv.imshow('mask',mask)
#     cv.imshow('res',res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv.destroyAllWindows()


# ## [imports]
# import cv2 as cv
# import sys
# ## [imports]
# ## [imread]
# img = cv.imread(cv.samples.findFile("starry_night.jpg"))
# ## [imread]
# ## [empty]
# if img is None:
#     sys.exit("Could not read the image.")
# ## [empty]
# ## [imshow]
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# ## [imshow]
# ## [imsave]
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img)
# ## [imsave]



# import cv2 as cv
# import numpy as np

# cam = cv.VideoCapture(0)

# if not cam.isOpened():
#     print('cannot open camera')
#     exit()

# while True:
#     ret,frame = cam.read()

#     if not ret:
#         print('cannot receive frame')
#         break

#     gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

#     cv.imshow('frame',gray)

#     if cv.waitKey(1) == ord('q'):
#         break
# cam.release()
# cv.destroyAllWindows()


import numpy as np
import cv2 as cv

cap = cv.VideoCapture('output.avi')

while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv

# cap = cv.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     frame = cv.flip(frame, 0)

#     # write the flipped frame
#     out.write(frame)

#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'):
#         break

# # Release everything if job is finished
# cap.release()
# out.release()
# cv.destroyAllWindows()
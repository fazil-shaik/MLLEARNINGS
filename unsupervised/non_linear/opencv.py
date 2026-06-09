import cv2

image = cv2.imread('images.jpeg',cv2.IMREAD_GRAYSCALE)

print(image.shape)

print(image[0,0])

image = image*2
cv2.imshow("cat",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

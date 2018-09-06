import cv2


def sketch(image):
    # convert image to gray scale:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # clean up image with gaussian blur:
    img_gray_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # extract edges:
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    # do a binary invert of image:
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Live Sketch WebCam', sketch(frame))
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

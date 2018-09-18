import cv2
import dlib
import numpy as np
import sys
from time import sleep
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17,68))
MOUTH_POINTS = list(range(48,61))
RIGHT_BROW_POINTS = list(range(17,22))
LEFT_BROW_POINTS = list(range(22,27))
RIGHT_EYE_POINTS = list(range(36,42))
LEFT_EYE_POINTS = list(range(42,48))
NOSE_POINTS = list(range(27,35))
JAW_POINTS = list(range(0,17))
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS]
COLOUR_CORRECT_BLUR_FRAC = 0.6
cascade_path = 'haar_cascades\haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im, dlibOn):
    if dlibOn == True:
        rects = detector(im, 1)
        if len(rects) > 1:
            return 'error'
        if len(rects) == 0:
            return 'error'
        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    else:
        rects = cascade.detectMultiScale(im, 1.3, 5)
        if len(rects) > 1:
            return 'error'
        if len(rects) == 0:
            return 'error'
        x,y,w,h = rects[0]
        rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0,0], point[0,1])
        cv2.putText(im, str(idx),pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0,255,255))
    return im

def draw_convex_hull(im, point, color):
    points = cv2.convexHull(point)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)
    im = np.array([im,im,im]).transpose((1,2,0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT),0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im

def transformation_from_points(points1,points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    C1 = np.mean(points1, axis=0)
    C2 = np.mean(points2, axis=0)
    points1 -= C1
    points2 -= C2
    S1 = np.std(points1)
    S2 = np.std(points2)
    points1 /= S1
    points2 /= S2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((S2/S1)*R, C2.T - (S2/S1)*R * C1.T)), np.matrix([0.,0.,1.])])

def read_im_and_landmarks(filter_name):
    im = cv2.imread('G:\Python Folder\Open-CV\jack.jpg')
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im, dlibOn)
    return im,s

def warp_im(im,M,dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amt = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amt = int(blur_amt)
    if blur_amt % 2 == 0:
        blur_amt += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amt, blur_amt), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amt, blur_amt), 0)
    
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64)/ im2_blur.astype(np.float64))

def face_swap(img, name):
    s = get_landmarks(img, dlibOn)
    if(s=='error'):
        print('No or Too many Faces')
        return img
    im1, landmarks1 = img,s
    im2, landmarks2 = read_im_and_landmarks(name)
    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1,landmarks1), warped_mask], axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')
    frame = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return image

cap = cv2.VideoCapture(0)
filter_image = cv2.imread('G:\Python Folder\Open-CV\miley.jpg')
dlibOn = False

while True:
    ret, frame = cap.read()
    if ret == 1:
        frame = cv2.resize(frame, None,fx=0.75,fy=0.75,interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame,1)
        swap = face_swap(frame, filter_image)
        cv2.imshow('Live Face Swap App', swap)
        if cv2.waitKey(1) == 13:
            break
    else:
        print('Somethings wrong')
        break
cap.release()    
cv2.destroyAllWindows()

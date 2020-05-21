import cv2
import numpy as mp
import dlib
from math import hypot
from playsound import playsound


# when EYEAR_THRESH is crossed, sound_alarm turns on the alarm
def sound_alarm():
    playsound('D:/Projects/Drowsiness Detection/gogo.wav')


EYEAR_THRESH = 5
EYEAR_CONSEC_FRAMES = 4

MOUTHAR_THRESH= 1.9
MOUTHAR_CONSEC_FRAMES = 2


# initialize the frame counter
COUNTER = 0
COUNTERT= 0
# indicate if the alarm is going off
ALARM_ON = False


# calculates blinking ratio of eye
def aspect_ratio(eye_points,facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    horizontal = cv2.line(frame, left_point, right_point, (255, 0, 0), 2)
    # print(landmarks.part(36)) #print 36th value

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    verticle = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    verticle_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    # print(verticle_length)# lenth of verticle line
    horizontal_lenth = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    # print(horizontal_lenth)# lenth of horizontal line

    aspect_ratio = horizontal_lenth / verticle_length
    return aspect_ratio


# to get the mid point of two points
def midpoint(p1,p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)


# captures Video
cap = cv2.VideoCapture(0)


# gives four face coordinates
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_COMPLEX

# loop will run until Escape(27) is pressed
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # print(faces) gives four face coordinates
        # x,y = face.left(),face.top()
        # x1,y1 = face.right(),face.bottom()
        # cv2.rectangle(frame, (x,y), (x1,y1), (255,0,0),2) #Draws the rectangle
        # predictor is the object that is going to find the landmarks

        landmarks = predictor(gray, face)

        left_eye_ratio = aspect_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio = aspect_ratio([42,43,44,45,46,47],landmarks)
        mouth_ratio = aspect_ratio([48,50,52,54,56,58],landmarks)

        avg_eye_ar = (left_eye_ratio + right_eye_ratio) / 2
        #print(avg_eye_ar)
        print(mouth_ratio) #open-small

        # for eyes
        if avg_eye_ar > 5:
            cv2.putText(frame,"Blinked",(500,30),font,0.5,(0,255,0))

        if avg_eye_ar > EYEAR_THRESH:
            COUNTER += 1
            # if the eyes were closed for a sufficient number of frames
            # then turns on the alarm
            if COUNTER >= EYEAR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    sound_alarm()
                # alert signal
                cv2.putText(frame, "ALERT!", (250, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

            # if avg_eye_ar is not above the threshold, reset the counter
        else:
            COUNTER = 0
            ALARM_ON = False


        # for mouth
        if mouth_ratio < 1.9:
            cv2.putText(frame,"Yawned",(350,30),font,0.5,(0,255,0))

        if mouth_ratio < MOUTHAR_THRESH:
            COUNTERT += 1
            # if the mouth was open for a sufficient number of frames
            # then turns on the alarm
            if COUNTERT >= MOUTHAR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    sound_alarm()
                # alert signal
                cv2.putText(frame, "ALERT!", (250, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

            # if avg_eye_ar is not above the threshold, reset the counter
        else:
            COUNTERT = 0
            ALARM_ON = False

    # Shows on the screen
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1)
    # key 27 is Esc
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

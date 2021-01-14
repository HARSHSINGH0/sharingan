import cv2 as cv
cv2=cv
import dlib

cap=cv.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def rescaleFrame(frame,scale=1):
    width=int(frame.shape[1]*scale)#frame.shape[1] is width of image
    height=int(frame.shape[0]*scale)#frame.shape[0] is height of image
    dimension=(width,height)
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)
def midlinepoint(p1,p2):
    return int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)
def eyetrack():
    while True:
        _,frame=cap.read()
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        gray=rescaleFrame(gray)
        frame=rescaleFrame(frame)
        faces=detector(gray)
        for face in faces:
            #print(face)
            

            
            landmarks=predictor(gray,face)
            left_point=(midlinepoint(landmarks.part(37),landmarks.part(41)))
            right_point=(midlinepoint(landmarks.part(38),landmarks.part(40)))
            hor_line=cv.line(frame,left_point,right_point,(0,0,255),1)
            up_point=(midlinepoint(landmarks.part(37),landmarks.part(38)))
            down_point=(midlinepoint(landmarks.part(41),landmarks.part(40)))
            ver_line=cv.line(frame,up_point,down_point,(0,0,255),1)

            left_point=(midlinepoint(landmarks.part(43),landmarks.part(47)))
            right_point=(midlinepoint(landmarks.part(44),landmarks.part(46)))
            hor_line=cv.line(frame,left_point,right_point,(0,0,255),1)
            up_point=(midlinepoint(landmarks.part(43),landmarks.part(44)))
            down_point=(midlinepoint(landmarks.part(47),landmarks.part(46)))
            ver_line=cv.line(frame,up_point,down_point,(0,0,255),1)


        cv.imshow("frame",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
eyetrack()
cap.release()
cv2.destroyAllWindows()
import cv2
import matplotlib.pyplot as plt

facecascade = cv2.CascadeClassifier(r"C:\Users\Akanksha\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalcatface.xml")
eyecascade = cv2.CascadeClassifier(r"C:\Users\Akanksha\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml")
test=facecascade.load(r"C:\Users\Akanksha\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalcatface.xml")
print(test)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, facecascade, eyecascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, facecascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = draw_boundary(roi_img, eyecascade, 1.1, 12, color['red'], "Eye")
        
    return img


# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Reading image from video stream
    _,img = video_capture.read()
    img = detect(img,facecascade, eyecascade)
    cv2.imshow("face detection",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
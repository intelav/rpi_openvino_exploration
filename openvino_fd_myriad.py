import cv2 as cv

net = cv.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')

net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

frame = cv.imread('/home/pi/build/armv7l/Release/cam2.jpg')

if frame is None:
    raise Exception('Image not found!')



blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)

out = net.forward()

for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    if confidence > 0.5:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
cv.imwrite('out.png', frame)

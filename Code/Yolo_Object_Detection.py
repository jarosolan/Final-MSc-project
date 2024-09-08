import cv2
import numpy as np
import pandas as pd
import glob
import random
import math
from collections import namedtuple

# Load Yolo
modelConfiguration = "C:/Users/Peter/Documents/Ida/training/yolo_custom_detection/yolov3_testing.cfg"
modelWeights = "C:/Users/Peter/Documents/Ida/training/yolo_custom_detection/yolov3_training_last.weights"
net = cv2.dnn.readNet(modelConfiguration, modelWeights)

# Name custom object
classes = ["Tree"]

# Images path
images_path = glob.glob(r"C:\Users\Peter\Documents\Ida\valid_b\*.jpg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#colors = np.random.uniform(0, 255, size=(len(classes), 3))
color = (255, 255, 255)

# Define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

# Insert here the path of your images #random.shuffle(images_path)
# to store all coordinates
data = [] 
det_array = []

# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1, fy=1) # here manipulate size of display
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    mAP_df = pd.DataFrame(columns=['Confidences', 'IoU', 'TP', 'FP', 'Acc TP', 'Acc FP', 'Precision', 'Recall'])
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25:
                # Object detected
                #print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Predicted coordinates
                plwa = x
                plha = y
                plwb = x + w
                plhb = y + h

                # Ground true coordinates
                tlwa = int(0.25*width)
                tlha = int(0.25*height)
                tlwb = int(math.ceil(0.75*width))
                tlhb = int(math.ceil(0.75*height))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                pred_pixels = [plwa, plha, plwb, plhb]
                gt_pixels = [tlwa, tlha, tlwb, tlhb]
                det = Detection(img_path, gt_pixels, pred_pixels)
                det_array.append(det)

    def bb_intersection_over_union(boxA, boxB):
       	# determine the (x, y)-coordinates of the intersection rectangle
       	xA = max(boxA[0], boxB[0])
       	yA = max(boxA[1], boxB[1])
       	xB = min(boxA[2], boxB[2])
       	yB = min(boxA[3], boxB[3])
       	# compute the area of intersection rectangle
       	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
       	# compute the area of both the prediction and ground-truth
       	# rectangles
       	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
       	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
       	# compute the intersection over union by taking the intersection
       	# area and dividing it by the sum of prediction + ground-truth
       	# areas - the interesection area
       	iou = interArea / float(boxAArea + boxBArea - interArea)
       	# return the intersection over union value
       	return iou                       
      
    for detection in det_array:
    # load the image
    # compute the intersection over union and display it
        iou = bb_intersection_over_union(detection.gt, detection.pred)
    #print("{}: {:.4f}".format(img_path, iou))
        #coor.append([img_path, detection.gt, detection.pred, max(confidences), iou])
        # if not cv2.imwrite('./IoU/{}'.format(detection.image_path), image):
        #    raise Exception("Could not write image")
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    print(indexes)
    #font = cv2.FONT_HERSHEY_PLAIN
    
    for i in range(len(boxes)):
        if i in indexes:
            #x, y, w, h = boxes[i]
            #label = str(classes[class_ids[i]])
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #idx = np.argsort(confidences[0])[::-1][0]
            data.append([img_path[-11:-4], plwa, plha, plwb, plhb, tlwa, tlha, tlwb, tlhb, max(confidences), iou])
            #text = "{}, {:.1f}%".format(classes[idx], 100*(max(confidences)))
            #cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            #print(idx)
    else:
        data.append([img_path[-11:-4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
    cv2.imshow("Image", img)   
    key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Translating data to dataframe and removing duplicates
df = pd.DataFrame(data)
# Renaming columns
df.columns = ['name', 'plwa', 'plha', 'plwb', 'plhb', 'tlwa', 'tlha', 'tlwb',
              'tlhb', 'con', 'iou']
df = df.drop_duplicates(subset=['name'], keep='first')
# Return and view dataframe without duplicates
print(df)

df = df.sort_values(by='con', ascending=False)

# mAP calcualation:
def calc_TP(row):
    iou = row['iou']
    if iou >= 0.5:
    	result = 1
    else:
    	result = 0
    return result
    
def calc_FP(row):
	iou = row['iou']
	if iou < 0.5:
		result = 1
	else:
		result = 0
	return result

df['TP'] = df.apply(calc_TP, axis=1)
df['FP'] = df.apply(calc_FP, axis=1)
df['Acc TP'] = df['TP'].cumsum(axis=0)
df['Acc FP'] = df['FP'].cumsum(axis=0)

def calc_Acc_Precision(row):
    precision = row['Acc TP'] / (row['Acc TP'] + row['Acc FP'])
    return precision
    
def calc_Acc_Recall(row):
    recall = row['Acc TP'] / (df.count()[0])
    return recall
    
df['Precision'] = df.apply(calc_Acc_Precision, axis=1)
df['Recall'] = df.apply(calc_Acc_Recall, axis=1)
    
import matplotlib.pyplot as plt
df.plot(kind='line',x='Recall',y='Precision',color='red')
plt.show()

def calc_PR_AUC(x, y):
   sm = 0
   for i in range(1, len(x)):
       h = x[i] - x[i-1]
       sm += h * (y[i-1] + y[i]) / 2
   return sm

calc_PR_AUC(df['Recall'], df['Precision'])

# Saving to csv file
df.to_csv("output_combined.csv")

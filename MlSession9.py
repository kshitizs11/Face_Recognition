import numpy as np
import cv2
import os

dataset_path = "./data/"

face_data = []
labels = []

for fx in os.listdir(dataset_path):
	if fx.endswith(".npy"):
		l = fx.split(".")[0]

		face_item = np.load(dataset_path+fx)
		print(face_item.shape)
		print(l)

		face_data.append(face_item)

		for i in range(face_item.shape[0]):
			labels.append(l)

x = np.concatenate(face_data,axis=0)
y = np.array(labels)

print(x.shape)
print(y.shape)

def distance(A,B):
	return np.sum((B-A)**2)**0.5

def kNN(X,Y,x_query,k=5):
	m = X.shape[0]
	distances = []
	for i in range(m):
		dis = distance(x_query,X[i])
		distances.append((dis,Y[i]))

	distances=sorted(distances)
	distances = distances[:k]
	distances = np.array(distances)
	labels = distances[:, 1]
	uniq_label,counts = np.unique(labels,return_counts=True)

	pred = uniq_label[counts.argmax()]
	return pred


#TEST FACE RECOG

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
	red,frame = cam.read()
	if red==True:
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray_frame,1.3,5)


		for face in faces:
			A,B,h,w = face
			cv2.rectangle(frame,(A,B),(A+w,B+h),(255,0,0),2)

			offset=10
			face_section = frame[B-offset : B+h+offset , A-offset : A+w+offset]
			face_section = cv2.resize(face_section , (100 , 100))

			name = kNN(x,y, face_section.reshape(1,-1))
			cv2.putText(frame, name.title(), (A, B-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)

		cv2.imshow("Windows",frame)

		key = cv2.waitKey(1)

		if key == ord("q"):
			break

cam.release()
cv2.destroyAllWindows()












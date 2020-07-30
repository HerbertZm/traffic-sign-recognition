import json
import requests
import cv2
import glob
import numpy as np
from skimage import transform
from skimage import exposure

class_names = open("signnames.csv").read().strip().split("\n")[1:]
class_names = [l.split(",")[1] for l in class_names]

test_images = []
files = glob.glob("C:/virtualenvs/proyectoIA/traffic-sign-recognition/pruebas/*.PNG")
for myFile in files:
    image = cv2.imread(myFile)
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    image = image.astype("float32") / 255.0
    image = image.tolist()
    test_images.append(image)

test_array = np.array(test_images).tolist()
data = json.dumps({"signature_name": "serving_default", "instances": np.array(test_array).tolist()})
#print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

headers = {"content-type": "application/json"}
json_response = requests.post('http://scr-rest.herokuapp.com/v1/models/trafficsignnet:predict', data=data, headers=headers)
#print(json_response.text)
predictions = json.loads(json_response.text)['predictions']

for i in range(0,68):
    print('The model thought this was a {} (class {})'.format(class_names[np.argmax(predictions[i])], np.argmax(predictions[i])))
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:39:59 2024

@author: gielo
"""

import requests
import base64

url = 'http://127.0.0.1:5000/predict'

with open("test_data/scissors.jpg", "rb") as image_file:
    image = image_file.read()

encoded_image = base64.encodebytes(image).decode('utf-8')

    
myobj = {'image_data': str(encoded_image)}

x = requests.post(url, json = myobj)

print(x.text)
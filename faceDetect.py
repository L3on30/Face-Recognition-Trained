from operator import index
from tracemalloc import stop
import tensorflow.keras
from PIL import Image , ImageOps
import numpy as np
import sys
import cv2
import os
import playsound
import wikipedia
from gtts import gTTS
from webdriver_manager.chrome import ChromeDriverManager

wikipedia.set_lang('vi')
language = 'en'
path = ChromeDriverManager().install()

def speak(text):
    tts = gTTS ( text = text, lang = language, slow = False )
    tts.save ("sound.mp3")
    playsound.playsound ("sound.mp3", True )
    os.remove ("sound.mp3 ")


cam = cv2.VideoCapture(0)
out = 0

def capture_image():
    ret, frame = cam.read()
    cv2.imwrite('img_detect.png', frame)
    


def face_detection() :
    # Disable scientific notation for clarity
    np.set_printoptions( suppress = True )

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into thekeras model
    data = np . ndarray(shape =(1 , 224 , 224 , 3), dtype = np.float32)
    # Replace this with the path to your image
    image = Image.open('img_detect.png')


    size = (224 , 224)
    image = ImageOps.fit( image , size , Image.ANTIALIAS )

# turn the image into a numpy array
    image_array = np.asarray(image)

# display the resized image
    image.show()

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    name = ["Sir", "stranger"]
    index = -1
    max_value = -1
    
    global out
    for i in range (0 , len(prediction[0])):
        if max_value < prediction[0][i]:
            max_value = prediction[0][i]
            index = i
    
    speak("Hello" + name[index])
    if name[index == "stranger"]:
        out = out + 1
    if out == 3: 
        sys.exit()
        #time.sleep(10)

while True:
    capture_image()
    face_detection()
cv2.destroyAllWindows()
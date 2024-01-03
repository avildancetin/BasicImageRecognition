from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

loaded_model = load_model('image_recognition_model.h5')

img_path = 'test_image_name.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = loaded_model.predict(img_array)
if prediction[0][0] >= 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")

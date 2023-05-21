import cv2
import numpy as np
import tensorflow as tf


vid=cv2.VideoCapture(0)

model=tf.keras.models.load_model("keras_model.h5")
while True:
    succes,img=vid.read()

    image=cv2.resize(img,(224,224))
    test_image = np.array(image, dtype=np.float32) 
    test_image = np.expand_dims(test_image, axis=0) 
    # 3. Normalizar la imagen 
    normalised_image = test_image/255.0

    #Resultado de predeccion
    prediction=model.predict(normalised_image) 
    print("prediction:", prediction)

    cv2.imshow("video",img)

    key = cv2.waitKey(1)
    if key == 27:
        print("Detenido")
        break

vid.release()
cv2.destroyAllWindows()   








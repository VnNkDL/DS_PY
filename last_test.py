from keras.models import Model, model_from_json
from keras.preprocessing.image import image_utils
import numpy as np
import matplotlib.pyplot as plt
import os
# Загружаем данные об архитектуре сети из файла json
json_file = open("save//model_29112022_114545.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("save//model_cnn.h5")
loaded_model.summary()
answer_model = Model(inputs=loaded_model.input, outputs = loaded_model.output)

for file in os.listdir('last_test'):
    img = image_utils.load_img('last_test//'+file, target_size=(150, 150))
    plt.imshow(img)
    img_array = image_utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    answer = answer_model.predict(img_array)
    print(answer)
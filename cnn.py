from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from datetime import datetime
import onnxmltools
#Каталоги
data_dir = 'images'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

epochs = 5
batch_size = 30
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

nb_train_samples = 3430
nb_validation_samples = 428
nb_test_samples = 432

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255,vertical_flip = True, horizontal_flip = True)

train_generator = datagen.flow_from_directory(
    train_dir,
    #save_to_dir = 'save//train',
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

val_generator = datagen.flow_from_directory(
    val_dir,
    #save_to_dir = 'save//val',
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

test_generator = datagen.flow_from_directory(
    test_dir,
    #save_to_dir = 'save//test',
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs=epochs,
    validation_data = val_generator,
    validation_steps = nb_validation_samples // batch_size)


scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

string = datetime.now().strftime("%d%m%Y_%H%M%S")

model_json = model.to_json()
json_file = open('save//model_{}.json'.format(string), 'w')
json_file.write(model_json)
json_file.close()

onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, 'save//model_{}.onnx'.format(string))

model.save_weights('save//model_cnn.h5')

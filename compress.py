# Konversi model Keras ke TensorFlow Lite (sekali saja, tidak perlu di server)
import tensorflow as tf

model = tf.keras.models.load_model("models/model_dnn.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Simpan model
with open("models/model_dnn.tflite", "wb") as f:
    f.write(tflite_model)

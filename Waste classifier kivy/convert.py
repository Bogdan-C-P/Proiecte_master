import tensorflow as tf

model = tf.keras.models.load_model("model_res152.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model_152.tflite', 'wb') as f:
  f.write(tflite_model)
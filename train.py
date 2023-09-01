import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import os

print(f"TensorFlow version = {tf.__version__}\n")

# Set a fixed random seed value, for reproducibility, this will allow us to get
# the same random numbers each time the notebook is run
SEED = 1024
np.random.seed(SEED)
tf.random.set_seed(SEED)

# the list of axis that data is available for
AXIS_LABELS = [
    "yaw",
    "pitch",
    "roll",
]

# The sample length to use for training and detection
SAMPLE_WINDOW = 20

NUM_AXIS = len(AXIS_LABELS)

# create a one-hot encoded matrix that is used in the output
ONE_HOT_ENCODED_AXIS = np.eye(NUM_AXIS)

inputs = []
outputs = []

df = pd.read_csv("imu.csv")

# read each csv file and push an input and output
for gesture_index in range(NUM_AXIS):
   gesture = AXIS_LABELS[gesture_index]
   print(f"Processing index {gesture_index} for gesture '{gesture}'.")

   output = ONE_HOT_ENCODED_AXIS[gesture_index]
   
   gesture_df = df[df['target_name'] == gesture]
   gesture_df.index = range(len(gesture_df.index))

   num_recordings = int(gesture_df.shape[0] / SAMPLE_WINDOW)

   print(f"\tThere are {num_recordings} recordings of the {gesture} gesture.")

   for i in range(num_recordings):
      tensor = []
      # this gives us an input tensor of SAMPLE_WINDOW x 3
      for j in range(SAMPLE_WINDOW):
         index = i * SAMPLE_WINDOW + j
         # normalize the input data, between 0 to 1.
         # This is roughly done. Not specific to each axis.
         tensor += [
               (gesture_df['x'][index] + 90) / 180,
               (gesture_df['y'][index] + 90) / 180,
               (gesture_df['z'][index] + 90) / 180
         ]

      inputs.append(tensor)
      outputs.append(output)

# convert the list to numpy array
inputs = np.array(inputs)
outputs = np.array(outputs)

print("Data set parsing and preparation complete.")

num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

# Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
inputs = inputs[randomize]
outputs = outputs[randomize]

# Split the recordings (group of samples) into three sets: training, testing and validation
TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])
print("Data set randomization and splitting complete.")

# build the model and train it
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu')) # relu is used for performance
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_AXIS, activation='softmax')) # softmax is used, because we only expect one gesture to occur per input
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(inputs_train, outputs_train, epochs=200, batch_size=1, validation_data=(inputs_validate, outputs_validate))

# increase the size of the graphs
plt.rcParams["figure.figsize"] = (20,10)

# graph the loss, the model above is configure to use "mean squared error" as the loss function
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("loss.png")

# use the model to predict the test inputs
predictions = model.predict(inputs_test)

print("input =\n", inputs_test)
# print the predictions and the expected ouputs
print("predictions =\n", np.round(predictions, decimals=3))
print("actual =\n", outputs_test)

# convert from [[0,1,0],...] format to a 1D array of indicies [1,2,0,1...] 
predicted_label = np.zeros(len(predictions))
actual_label = np.zeros(len(outputs_test))
for i in range(0, len(predictions)):
   predicted_label[i] = np.argmax(predictions[i])
   actual_label[i] = np.argmax(outputs_test[i])

# Plot the predictions along with to the test data
plt.clf()
confusion_matrix = tf.math.confusion_matrix(actual_label, predicted_label, num_classes=3)
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig("matrix.png")

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("movement_model.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize("movement_model.tflite")
print("Model is %d bytes" % basic_model_size)
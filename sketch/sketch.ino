#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_HMC5883_U.h>

#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

const int numSamples = 20;

int samplesRead = 0;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "yaw",
  "pitch",
  "roll"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// Assign a unique ID to this sensor at the same time
Adafruit_HMC5883_Unified mag = Adafruit_HMC5883_Unified(12345);

#include "model.h"


void setup() {

  Serial.begin(9600);
  Serial.println("HMC5883 Magnetometer Test"); Serial.println("");
  
  //Initialise the sensor
  if(!mag.begin())
  {
    //There was a problem detecting the HMC5883 ... check your connections
    Serial.println("Ooops, no HMC5883 detected ... Check your wiring!");
    while(1);
  }

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {

  sensors_event_t event; 
  mag.getEvent(&event);

  //normalise the data
  tflInputTensor->data.f[samplesRead * 3 + 0] = (event.magnetic.x + 90.0) / 180.0;
  tflInputTensor->data.f[samplesRead * 3 + 1] = (event.magnetic.y + 90.0) / 180.0;
  tflInputTensor->data.f[samplesRead * 3 + 2] = (event.magnetic.z + 90.0) / 180.0;

  samplesRead++;

  //once we have gather enough data for the input tensor we can start inferencing
  if (samplesRead == numSamples) {

    long int start_time = micros();

    TfLiteStatus invokeStatus = tflInterpreter->Invoke();

    Serial.print("Inference time (us): ");
    Serial.println(micros() - start_time);

    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }

    for (int i = 0; i < NUM_GESTURES; i++) {
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 6);
    }
    Serial.println();
    samplesRead = 0;
    delay(1000);
  }

}
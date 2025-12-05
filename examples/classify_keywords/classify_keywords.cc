// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Keyword spotting example for "left", "right", "go" navigation commands
// Based on the classify_speech example, adapted for custom trained model

#include <cmath>

#include "libs/audio/audio_service.h"
#include "libs/base/filesystem.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/audio_models.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/tflite-micro/tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {

// Model configuration
constexpr int kTensorArenaSize = 1 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

// Audio configuration
constexpr int kNumDmaBuffers = 2;
constexpr int kDmaBufferSizeMs = 50;
constexpr int kDmaBufferSize = kNumDmaBuffers *
                               tensorflow::kKeywordDetectorSampleRateMs *
                               kDmaBufferSizeMs;
constexpr int kAudioServicePriority = 4;
constexpr int kDropFirstSamplesMs = 150;

AudioDriverBuffers<kNumDmaBuffers, kDmaBufferSize> audio_buffers;
AudioDriver audio_driver(audio_buffers);

constexpr int kAudioBufferSizeMs = tensorflow::kKeywordDetectorDurationMs;
constexpr int kAudioBufferSize =
    kAudioBufferSizeMs * tensorflow::kKeywordDetectorSampleRateMs;

// Model and inference configuration
constexpr float kThreshold = 0.50;  // Confidence threshold (50%)
constexpr int kTopK = 3;           // Show top 3 predictions
constexpr char kModelName[] = "/models/model_int8_edgetpu.tflite";

// Audio input buffer
std::array<int16_t, tensorflow::kKeywordDetectorAudioSize> audio_input;

// Label names for our 4-class model
const char* labels[] = {
    "left",     // 0
    "right",    // 1
    "go",       // 2
    "unknown"   // 3 (includes silence)
};
constexpr int kNumLabels = 4;

// Run inference on the audio input
void run(tflite::MicroInterpreter* interpreter, FrontendState* frontend_state) {
  auto input_tensor = interpreter->input_tensor(0);

  // Calculate audio loudness (RMS level)
  int64_t sum_squares = 0;
  for (size_t i = 0; i < audio_input.size(); ++i) {
    int32_t sample = audio_input[i];
    sum_squares += sample * sample;
  }
  float rms = sqrtf(static_cast<float>(sum_squares) / audio_input.size());
  float loudness_db = 20.0f * log10f(rms / 32768.0f + 1e-10f);  // dB relative to full scale

  // Preprocess audio into spectrogram features
  auto preprocess_start = TimerMillis();
  tensorflow::KeywordDetectorPreprocessInput(audio_input.data(), input_tensor,
                                             frontend_state);
  FrontendReset(frontend_state);
  auto preprocess_end = TimerMillis();

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("ERROR: Failed to invoke inference\r\n");
    return;
  }

  auto inference_end = TimerMillis();

  // Get output tensor
  auto output_tensor = interpreter->output_tensor(0);

  // Find the class with highest score
  int8_t max_score = -128;
  int max_index = -1;

  for (int i = 0; i < kNumLabels; i++) {
    int8_t score = output_tensor->data.int8[i];
    if (score > max_score) {
      max_score = score;
      max_index = i;
    }
  }

  // Convert int8 score to probability (0-255 range)
  uint8_t confidence = max_score + 128;
  float probability = confidence / 255.0f;

  // Always show prediction for debugging
  printf("Prediction: %s (%.2f%%) | ", labels[max_index], probability * 100.0f);

  // Show all class scores
  printf("Scores: ");
  for (int i = 0; i < kNumLabels; i++) {
    uint8_t score = output_tensor->data.int8[i] + 128;
    printf("%s:%.1f%% ", labels[i], (score / 255.0f) * 100.0f);
  }
  printf("| ");

  // Highlight if above threshold and is a keyword
  if (probability > kThreshold && max_index >= 0 && max_index <= 2) {
    printf("*** DETECTED: %s ***", labels[max_index]);
  }

  // Print timing and audio level information
  printf("Audio level: %.1f dB | RMS: %.0f | Preprocess: %lums, Inference: %lums\r\n",
         loudness_db,
         rms,
         static_cast<uint32_t>(preprocess_end - preprocess_start),
         static_cast<uint32_t>(inference_end - preprocess_end));
}

}  // namespace

[[noreturn]] void Main() {
  printf("\r\n");
  printf("========================================\r\n");
  printf("Keyword Detector: left, right, go\r\n");
  printf("========================================\r\n");
  printf("\r\n");

  // Load the trained model
  std::vector<uint8_t> model_data;
  if (!LfsReadFile(kModelName, &model_data)) {
    printf("ERROR: Failed to load model from %s\r\n", kModelName);
    printf("Please ensure your trained model is copied to /models/\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Model loaded successfully (%zu bytes)\r\n", model_data.size());

  // Get Edge TPU context
  auto edgetpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!edgetpu_context) {
    printf("ERROR: Failed to initialize Edge TPU\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Edge TPU initialized\r\n");

  // Setup TFLite interpreter
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model_data.data()),
                                       resolver, tensor_arena, kTensorArenaSize,
                                       &error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: Failed to allocate tensors\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Tensors allocated\r\n");

  // Print model input/output information
  auto* input = interpreter.input_tensor(0);
  auto* output = interpreter.output_tensor(0);
  printf("\r\n");
  printf("Model Info:\r\n");

  // Print actual tensor dimensions
  printf("  Model expects input shape: ");
  if (input->dims->size == 2) {
    printf("[%d, %d] = %d features\r\n",
           input->dims->data[0], input->dims->data[1],
           input->dims->data[1]);
  } else if (input->dims->size == 3) {
    printf("[%d, %d, %d] = %d features\r\n",
           input->dims->data[0], input->dims->data[1], input->dims->data[2],
           input->dims->data[1] * input->dims->data[2]);
  }

  printf("  KeywordDetector preprocessing produces: [1, 198, 32] = 6336 features\r\n");
  printf("  MISMATCH: Model trained with [1, 49, 40] = 1960 features!\r\n");
  printf("\r\n");
  printf("  Input type: %s\r\n",
         input->type == kTfLiteInt8 ? "INT8" : "UNKNOWN");
  printf("  Output shape: [%d, %d]\r\n",
         output->dims->data[0], output->dims->data[1]);
  printf("  Output type: %s\r\n",
         output->type == kTfLiteInt8 ? "INT8" : "UNKNOWN");
  printf("  Number of classes: %d\r\n", kNumLabels);
  printf("\r\n");
  printf("WARNING: Preprocessing mismatch detected!\r\n");
  printf("The model will receive wrong-shaped input and produce garbage output.\r\n");
  printf("You need to retrain with KeywordDetector-compatible parameters:\r\n");
  printf("  - 2 seconds audio (not 1 second)\r\n");
  printf("  - 25ms window, 10ms stride (not 30ms/20ms)\r\n");
  printf("  - 32 mel bins (not 40)\r\n");
  printf("\r\n");

  // Setup audio frontend (spectrogram preprocessing)
  FrontendState frontend_state{};
  if (!tensorflow::PrepareAudioFrontEnd(
          &frontend_state, tensorflow::AudioModel::kKeywordDetector)) {
    printf("ERROR: Failed to prepare audio frontend\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Audio frontend initialized\r\n");

  // Setup audio capture
  AudioDriverConfig audio_config{AudioSampleRate::k16000_Hz, kNumDmaBuffers,
                                 kDmaBufferSizeMs};
  AudioService audio_service(&audio_driver, audio_config, kAudioServicePriority,
                             kDropFirstSamplesMs);
  LatestSamples audio_latest(MsToSamples(
      AudioSampleRate::k16000_Hz, tensorflow::kKeywordDetectorDurationMs));
  audio_service.AddCallback(
      &audio_latest,
      +[](void* ctx, const int32_t* samples, size_t num_samples) {
        static_cast<LatestSamples*>(ctx)->Append(samples, num_samples);
        return true;
      });
  printf("Audio service started\r\n");
  printf("\r\n");
  printf("Listening for keywords: left, right, go...\r\n");
  printf("Speak clearly into the microphone.\r\n");
  printf("\r\n");

  // Wait for first audio buffers to fill
  vTaskDelay(pdMS_TO_TICKS(tensorflow::kKeywordDetectorDurationMs));

  // Main inference loop
  while (true) {
    // Get latest audio samples
    audio_latest.AccessLatestSamples(
        [](const std::vector<int32_t>& samples, size_t start_index) {
          size_t i, j = 0;
          // Circular buffer read: start_index to end
          for (i = 0; i < samples.size() - start_index; ++i) {
            audio_input[i] = samples[i + start_index] >> 16;
          }
          // Wrap around: beginning to start_index
          for (j = 0; j < samples.size() - i; ++j) {
            audio_input[i + j] = samples[j] >> 16;
          }
        });

    // Run inference
    run(&interpreter, &frontend_state);

    // Rate limit inference (500ms intervals for 75% overlap)
    // This ensures keywords spoken at window boundaries are captured
    vTaskDelay(pdMS_TO_TICKS(500));
  }
}

}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}

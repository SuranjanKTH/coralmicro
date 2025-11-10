# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Keyword spotting example for "left", "right", "go" commands
# Uses custom trained model from train_keyword_complete.ipynb

add_executable_m7(classify_keywords
    classify_keywords.cc
    DATA
    ${CMAKE_SOURCE_DIR}/models/model_int8_edgetpu.tflite
)

target_link_libraries(classify_keywords
    libs_base-m7_freertos
    libs_audio_freertos
)

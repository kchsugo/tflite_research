cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(CMAKE_CXX_STANDARD 17)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU ON CACHE BOOL "" FORCE)

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

add_executable(minimal minimal.cc)
target_link_libraries(minimal
  tensorflow-lite
  absl::log
  absl::log_internal_message
  absl::log_internal_nullguard
  absl::log_internal_check_op
  absl::log_internal_conditions
  absl::log_internal_format
  absl::log_internal_globals
  absl::log_internal_log_sink_set
  absl::status
  absl::strings
  absl::synchronization
  absl::time
)
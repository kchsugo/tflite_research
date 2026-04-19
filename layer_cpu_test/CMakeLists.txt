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

# ★ XNNPACK 비활성화 — 개별 op 프로파일링 위해 필수
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)

cmake_minimum_required(VERSION 3.16)
project(split_profiler C CXX)
set(CMAKE_CXX_STANDARD 17)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../tensorflow" ABSOLUTE)
endif()

# first/last_delegate_node_index 필드를 켜는 매크로 (add_subdirectory 전에).
add_compile_definitions(TFLITE_DEBUG_DELEGATE)

set(TFLITE_ENABLE_GPU ON CACHE BOOL "" FORCE)

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL)

set(TFLITE_LINK_LIBS
  tensorflow-lite
  absl::log absl::log_internal_message absl::log_internal_nullguard
  absl::log_internal_check_op absl::log_internal_conditions
  absl::log_internal_format absl::log_internal_globals
  absl::log_internal_log_sink_set
  absl::status absl::strings absl::synchronization absl::time)

add_executable(split_profiler split_profiler.cc)
if(MSVC)
  target_compile_options(split_profiler PRIVATE /utf-8)
endif()
target_link_libraries(split_profiler ${TFLITE_LINK_LIBS})

add_executable(jackson_online jackson_online.cc)
if(MSVC)
  target_compile_options(jackson_online PRIVATE /utf-8)
endif()
target_link_libraries(jackson_online ${TFLITE_LINK_LIBS})

# 통합 플래너 ({통짜-GPU, 통짜-CPU, 분할} per-task 배정 + 캘리브레이션)
add_executable(unified unified.cc)
if(MSVC)
  target_compile_options(unified PRIVATE /utf-8)
endif()
target_link_libraries(unified ${TFLITE_LINK_LIBS})

# 배칭 측정 (batch 차원으로 묶어 GPU 한 번에 → per-task 시간)
add_executable(batch_test batch_test.cc)
if(MSVC)
  target_compile_options(batch_test PRIVATE /utf-8)
endif()
target_link_libraries(batch_test ${TFLITE_LINK_LIBS})

cmake_minimum_required(VERSION 3.16)
project(multitask C CXX)
set(CMAKE_CXX_STANDARD 17)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Directory that contains the TensorFlow project")
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../tensorflow" ABSOLUTE)
endif()

# 이 도구는 GPU delegate 전체를 적용할 뿐 first_delegate_node_index를 쓰지 않으므로
# TFLITE_DEBUG_DELEGATE 매크로가 필요 없다. (internal_split_test와 다른 점)
set(TFLITE_ENABLE_GPU ON CACHE BOOL "" FORCE)

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL)

add_executable(multitask minimal.cc)

# MSVC(한국어 Windows, 코드페이지 949)에서 UTF-8 한글 주석이 뒤 코드를 삼키는 문제
# (warning C4819 → 파싱 오류) 방지: 소스를 UTF-8로 읽도록 명시.
if(MSVC)
  target_compile_options(multitask PRIVATE /utf-8)
endif()

set(TFLITE_LINK_LIBS
  tensorflow-lite
  absl::log absl::log_internal_message absl::log_internal_nullguard
  absl::log_internal_check_op absl::log_internal_conditions
  absl::log_internal_format absl::log_internal_globals
  absl::log_internal_log_sink_set
  absl::status absl::strings absl::synchronization absl::time)

target_link_libraries(multitask ${TFLITE_LINK_LIBS})

# 오프라인 프로파일러 (같은 링크 세트 재사용)
add_executable(profile profile.cc)
if(MSVC)
  target_compile_options(profile PRIVATE /utf-8)
endif()
target_link_libraries(profile ${TFLITE_LINK_LIBS})

# 온라인 실행기 (배분 + 실제 동시 실행 + 예측/실측 비교)
add_executable(online online.cc)
if(MSVC)
  target_compile_options(online PRIVATE /utf-8)
endif()
target_link_libraries(online ${TFLITE_LINK_LIBS})

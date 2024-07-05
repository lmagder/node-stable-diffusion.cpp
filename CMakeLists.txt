cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0091 NEW)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

#required for linux, harmless elsehwer
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

find_package(CUDAToolkit)

if (MSVC)
  set(CMake_MSVC_PARALLEL ON)
  set(MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)
endif()

FetchContent_Declare(
  stable-diffusion-cpp
  GIT_REPOSITORY https://github.com/lmagder/stable-diffusion.cpp.git
  GIT_TAG        83e80d5171f02b0f806794e0b0329828f87702f9 # fix-untyped-tensors
)

set(GGML_STATIC ON)

if (CUDAToolkit_FOUND)
  set(SD_CUBLAS ON)
  set(SD_FLASH_ATTN ON)
  set(GGML_CUDA_F16 ON)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
  endif()
  get_target_property(LIBCUDAPATH CUDA::cuda_driver IMPORTED_LOCATION)
  if (${LIBCUDAPATH} STREQUAL "LIBCUDAPATH-NOTFOUND")
    set(LIBCUDASTUB ${CMAKE_SYSROOT}/usr/local/cuda-${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/targets/x86_64-linux/lib/stubs/libcuda.so)
    if (EXISTS ${LIBCUDASTUB})
      message(STATUS "Can't find libcuda.so but have a valid toolkit. Asssuming we are using the stub")
      set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION ${LIBCUDASTUB})
    endif()
  endif()
endif()

FetchContent_MakeAvailable(stable-diffusion-cpp)

project(node-stable-diffusion-cpp)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CMAKE_JS_INC})
target_compile_definitions(${PROJECT_NAME} PRIVATE NAPI_VERSION=9 NODE_ADDON_API_DISABLE_DEPRECATED)

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} stable-diffusion)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})

  #copy cublas DLLs
  if (SD_CUBLAS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CUDAToolkit_TARGET_DIR}/bin/cublas64_${CUDAToolkit_VERSION_MAJOR}.dll $<TARGET_FILE_DIR:${PROJECT_NAME}>
      COMMAND_EXPAND_LISTS
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CUDAToolkit_TARGET_DIR}/bin/cublasLt64_${CUDAToolkit_VERSION_MAJOR}.dll $<TARGET_FILE_DIR:${PROJECT_NAME}>
      COMMAND_EXPAND_LISTS
    )
  endif()
endif()

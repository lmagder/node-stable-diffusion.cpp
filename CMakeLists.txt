cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0091 NEW)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

#required for linux, harmless elsehwer
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-expansion-to-defined)
endif()

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

FetchContent_Declare(
  implib-so
  GIT_REPOSITORY https://github.com/yugr/Implib.so.git
  GIT_TAG        f9ff166845a59327eda431af82ee85a9c7532c5d
)

set(IMPLIB_SOURCE_FILES "")
if (CUDAToolkit_FOUND)
  set(SD_CUBLAS ON)
  set(SD_FLASH_ATTN ON)
  set(GGML_CUDA_F16 ON)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
  endif()
  if (CMAKE_SYSTEM_NAME MATCHES "Linux")

    FetchContent_MakeAvailable(implib-so)
    FetchContent_GetProperties(implib-so SOURCE_DIR IMPLIB_SO_FOLDER)

    get_target_property(LIBCUDAPATH CUDA::cuda_driver IMPORTED_LOCATION)
    if (${LIBCUDAPATH} STREQUAL "LIBCUDAPATH-NOTFOUND")
      set(LIBCUDASTUB ${CMAKE_SYSROOT}/usr/local/cuda-${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/targets/x86_64-linux/lib/stubs/libcuda.so)
      if (EXISTS ${LIBCUDASTUB})
        message(STATUS "Can't find libcuda.so but have a valid toolkit. Asssuming we are using the stub")
        set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION ${LIBCUDASTUB})
      endif()
    endif()

    set(IMPLIB_SOURCE_FILES ${CMAKE_CURRENT_BINARY_DIR}/libnvcuda.so.tramp.S ${CMAKE_CURRENT_BINARY_DIR}/libnvcuda.so.init.c)

    add_custom_command(
      OUTPUT ${IMPLIB_SOURCE_FILES}
      COMMAND ${IMPLIB_SO_FOLDER}/implib-gen.py -q ${LIBCUDAPATH}
      DEPENDS ${LIBCUDAPATH}
    )
  endif()
endif()

FetchContent_MakeAvailable(stable-diffusion-cpp)

project(node-stable-diffusion-cpp)

enable_language(ASM)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")
add_library(node-stable-diffusion-cpp SHARED ${SOURCE_FILES} ${IMPLIB_SOURCE_FILES} ${CMAKE_JS_SRC})

set_target_properties(node-stable-diffusion-cpp PROPERTIES PREFIX "" SUFFIX ".node")
set_target_properties(node-stable-diffusion-cpp PROPERTIES CXX_STANDARD 20)

target_include_directories(node-stable-diffusion-cpp SYSTEM PRIVATE ${CMAKE_JS_INC})
target_compile_definitions(node-stable-diffusion-cpp PRIVATE NAPI_VERSION=9 NODE_ADDON_API_DISABLE_DEPRECATED)

target_link_libraries(node-stable-diffusion-cpp ${CMAKE_JS_LIB} stable-diffusion)

if (CUDAToolkit_FOUND)
  file(GENERATE OUTPUT $<TARGET_FILE_DIR:node-stable-diffusion-cpp>/cuda_version.json INPUT ${CUDAToolkit_LIBRARY_ROOT}/version.json)
else()
  file(GENERATE OUTPUT $<TARGET_FILE_DIR:node-stable-diffusion-cpp>/cuda_version.json CONTENT "{}")
endif()

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
  target_link_options(node-stable-diffusion-cpp PRIVATE "/DELAYLOAD:nvcuda.dll")
endif()



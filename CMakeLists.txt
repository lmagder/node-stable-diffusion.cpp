cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)
FetchContent_Declare(
  stable-diffusion-cpp
  GIT_REPOSITORY https://github.com/lmagder/stable-diffusion.cpp.git
  GIT_TAG        83e80d5171f02b0f806794e0b0329828f87702f9 # fix-untyped-tensors
)

set(SD_CUBLAS ON)
set(SD_FLASH_ATTN ON)
set(GGML_CUDA_F16 ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

FetchContent_MakeAvailable(stable-diffusion-cpp)

# Name of the project (will be the name of the plugin)
project(node-stable-diffusion.cpp)

# Build a shared library named after the project from the files in `src/`
file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# Gives our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# Essential include files to build a node addon,
# You should add this line in every CMake.js based project
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CMAKE_JS_INC})

target_compile_definitions(${PROJECT_NAME} PRIVATE NAPI_VERSION=8)

# Essential library files to link to a node addon
# You should add this line in every CMake.js based project
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB} stable-diffusion)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
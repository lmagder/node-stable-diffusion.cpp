cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

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

# Essential library files to link to a node addon
# You should add this line in every CMake.js based project
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

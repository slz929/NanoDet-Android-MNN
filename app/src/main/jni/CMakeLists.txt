project(nanodet-mnn)

cmake_minimum_required(VERSION 3.14.7)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-3.4.18-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230517-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# mnn
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/)
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/expr/)
add_library(libMNN STATIC IMPORTED)
add_library(libMNN_CL STATIC IMPORTED)
add_library(libMNN_Express STATIC IMPORTED)
set_target_properties(
        libMNN
        libMNN_CL
        libMNN_Express
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN.so
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_CL.so
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_Express.so
)

add_library(nanodetmnn SHARED jni_intereface.cpp NanoDet.cpp ndkcamera.cpp)

target_link_libraries(nanodetmnn ncnn ${OpenCV_LIBS} camera2ndk mediandk libMNN libMNN_CL libMNN_Express)
# ncnn: ndkcamera.cpp

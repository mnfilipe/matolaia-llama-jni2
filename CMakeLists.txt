cmake_minimum_required(VERSION 3.22)
project(matola_llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Só CPU, sem GPU/Vulkan/OpenCL — mantém o build simples e previsível.
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# O checkout do Actions vai clonar o llama.cpp para esta pasta (ver workflow).
add_subdirectory(llama.cpp)

add_library(matola_llama SHARED
    cpp/matola_llama_jni.cpp
)

target_include_directories(matola_llama PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
    llama.cpp/common
)

target_link_libraries(matola_llama
    llama
    common
    log
    android
)

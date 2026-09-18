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

# O checkout do Actions vai clonar o llama.cpp para esta pasta.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Ajustado para CPP em letras maiúsculas para bater com o seu repositório
add_library(matola_llama SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/CPP/matola_llama_jni.cpp
)

target_include_directories(matola_llama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
)

target_link_libraries(matola_llama
    llama
    common
    log
    android
)

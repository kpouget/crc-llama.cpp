cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

# function(detect_host_compiler)
#     find_program(HOST_C_COMPILER NAMES gcc clang NO_CMAKE_FIND_ROOT_PATH)
#     find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)

#     set(HOST_C_COMPILER "${HOST_C_COMPILER}" PARENT_SCOPE)
#     set(HOST_CXX_COMPILER "${HOST_CXX_COMPILER}" PARENT_SCOPE)
# endfunction()

message(STATUS "Enable API Remoting frontend found")

ggml_add_backend_library(ggml-remotingfrontend
                         ggml-remoting-frontend.cpp
                         ../../include/ggml-remoting-frontend.h
                        )

#target_link_libraries(ggml-remotingfrontend PRIVATE remotingfrontend)
target_include_directories(ggml-remotingfrontend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

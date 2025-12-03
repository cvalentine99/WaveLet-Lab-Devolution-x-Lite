cmake_minimum_required(VERSION 3.20)
project(rf_forensics_holoscan LANGUAGES CXX CUDA)

# C++17 required for Holoscan
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(holoscan REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)

# CUDA architectures (RTX 30xx = sm_86, RTX 40xx = sm_89)
set(CMAKE_CUDA_ARCHITECTURES 86 89)

# =============================================================================
# CUDA Kernels Library
# =============================================================================
add_library(rf_kernels STATIC
    src/kernels/psd_kernels.cu
    src/kernels/cfar_kernels.cu
)

target_include_directories(rf_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(rf_kernels PUBLIC
    CUDA::cufft
    CUDA::cudart
)

set_target_properties(rf_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# =============================================================================
# Holoscan Operators Library
# =============================================================================
add_library(rf_operators SHARED
    src/operators/usdr_source_op.cpp
    src/operators/psd_op.cpp
    src/operators/cfar_op.cpp
    src/operators/peak_op.cpp
)

target_include_directories(rf_operators PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/include
    /usr/local/include/usdr
)

target_link_libraries(rf_operators PUBLIC
    holoscan::core
    rf_kernels
    CUDA::cufft
    CUDA::cudart
)

# Link USDR if available
find_library(USDR_LIBRARY usdr HINTS /usr/local/lib /opt/usdr/lib)
if(USDR_LIBRARY)
    target_link_libraries(rf_operators PUBLIC ${USDR_LIBRARY})
    target_compile_definitions(rf_operators PRIVATE HAS_USDR=1)
    message(STATUS "Found USDR library: ${USDR_LIBRARY}")
else()
    message(WARNING "USDR library not found - using simulated source")
endif()

# =============================================================================
# Main Application
# =============================================================================
add_executable(rf_forensics_app
    src/app/rf_forensics_app.cpp
)

target_link_libraries(rf_forensics_app PRIVATE
    holoscan::core
    rf_operators
)

# =============================================================================
# Python Bindings (optional)
# =============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)

    if(pybind11_FOUND)
        pybind11_add_module(rf_forensics_holoscan python/bindings.cpp)
        target_link_libraries(rf_forensics_holoscan PRIVATE
            holoscan::core
            rf_operators
        )
        message(STATUS "Python bindings will be built")
    else()
        message(WARNING "pybind11 not found - Python bindings disabled")
    endif()
endif()

# =============================================================================
# Install
# =============================================================================
install(TARGETS rf_forensics_app rf_operators rf_kernels
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(TARGET rf_forensics_holoscan)
    install(TARGETS rf_forensics_holoscan
        LIBRARY DESTINATION lib/python
    )
endif()

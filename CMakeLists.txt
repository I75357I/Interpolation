cmake_minimum_required(VERSION 3.8)

# Enable hot-reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")
endif()

project("InterpolationLibrary")

if (CMAKE_VERSION VERSION_GREATER 3.12)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# --- 2D Interpolation Library ---
set(SRC_2D
    "src/2D Interpolation/InterpolatorCommon.h"
    "src/2D Interpolation/CubicSplineInterpolator.h"
    "src/2D Interpolation/CubicSplineInterpolator.cpp"
    "src/2D Interpolation/HermiteInterpolator.h"
    "src/2D Interpolation/HermiteInterpolator.cpp"
    "src/2D Interpolation/LinearInterpolator.h"
    "src/2D Interpolation/LinearInterpolator.cpp"
)

add_library(MultiInterpolation ${SRC_2D})
target_include_directories(MultiInterpolation PUBLIC "${CMAKE_SOURCE_DIR}/src/2D Interpolation")

# --- 3D Interpolation Library ---
set(SRC_3D
    "src/3D Interpolation/InterpolatorCommon3D.h"
    "src/3D Interpolation/Delaune.h"
    "src/3D Interpolation/Delaune.cpp"
    "src/3D Interpolation/DelauneInterpolation.h"
    "src/3D Interpolation/DelauneInterpolation.cpp"
    "src/3D Interpolation/NearestTriangularInterpolation3D.h"
    "src/3D Interpolation/NearestTriangularInterpolation3D.cpp"
)

add_library(ThreeDInterpolation ${SRC_3D})
target_include_directories(ThreeDInterpolation PUBLIC "${CMAKE_SOURCE_DIR}/src/3D Interpolation")

# --- NN Interpolation Library ---
set(SRC_NN
    "src/NN Interpolation/activation.h"
    "src/NN Interpolation/layer.h"
    "src/NN Interpolation/losses"
    "src/NN Interpolation/NeuralNetwork.h"
    "src/NN Interpolation/nn.h"
    "src/NN Interpolation/utils.h"
)

add_library(NNInterpolation ${SRC_3D})
target_include_directories(NNInterpolation PUBLIC "${CMAKE_SOURCE_DIR}/src/NN Interpolation")

# 2D Interpolation Tests
add_executable(Linear_test "tests/2DInterpolator/Linear_test.cpp")
target_include_directories(Linear_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Linear_test PRIVATE MultiInterpolation ThreeDInterpolation)

add_executable(Cubic_spline_test "tests/2DInterpolator/Cubic_spline_test.cpp")
target_include_directories(Cubic_spline_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Cubic_spline_test PRIVATE MultiInterpolation ThreeDInterpolation)

add_executable(Hermite_test "tests/2DInterpolator/Hermite_test.cpp")
target_include_directories(Hermite_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Hermite_test PRIVATE MultiInterpolation ThreeDInterpolation)

# 3D Interpolation Tests
add_executable(Delaune_test "tests/3DInterpolator/Delaune_test.cpp")
target_include_directories(Delaune_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Delaune_test PRIVATE MultiInterpolation ThreeDInterpolation)

add_executable(Nearest_Triangle_test "tests/3DInterpolator/Nearest_Triangle_test.cpp")
target_include_directories(Nearest_Triangle_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Nearest_Triangle_test PRIVATE MultiInterpolation ThreeDInterpolation)

add_executable(NN_test "tests/NN/NN_test.cpp")
target_include_directories(NN_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(NN_test PRIVATE MultiInterpolation NNInterpolation)

install(TARGETS Linear_test Cubic_spline_test Hermite_test Delaune_test Nearest_Triangle_test NN_test
        DESTINATION bin)

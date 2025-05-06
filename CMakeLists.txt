cmake_minimum_required(VERSION 3.8)

# Enable hot-reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")
endif()

project(InterpolationLibrary)

# Require C++17 on CMake ≥ 3.12
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
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

add_library(MultiInterpolation STATIC ${SRC_2D})
target_include_directories(MultiInterpolation PUBLIC
    "${CMAKE_SOURCE_DIR}/src/2D Interpolation"
)

# --- 3D Interpolation Library ---
set(SRC_3D
    "src/3D Interpolation/InterpolatorCommon3D.h"
    "src/3D Interpolation/Delaune_BW.h"
    "src/3D Interpolation/DelauneInterpolation.h"
    "src/3D Interpolation/DelauneInterpolation.cpp"
    "src/3D Interpolation/NearestTriangularInterpolation3D.h"
    "src/3D Interpolation/NearestTriangularInterpolation3D.cpp"
)

add_library(ThreeDInterpolation STATIC ${SRC_3D})
target_include_directories(ThreeDInterpolation PUBLIC
    "${CMAKE_SOURCE_DIR}/src/3D Interpolation"
)

# --- NN Interpolation Library (header-only) ---
add_library(NNInterpolation INTERFACE)
target_include_directories(NNInterpolation INTERFACE
    "${CMAKE_SOURCE_DIR}/src/NN Interpolation"
)

# --- MD Interpolation Library ---
set(SRC_MD
    "src/MD Interpolation/InterpolatorCommonMD.h"
    "src/MD Interpolation/Delaune_BW_MD.h"
    "src/MD Interpolation/DelauneInterpolation_MD.h"
    "src/MD Interpolation/NearestSimplexInterpolationMD.h"
)

add_library(MDInterpolation INTERFACE)
target_include_directories(MDInterpolation INTERFACE
    "${CMAKE_SOURCE_DIR}/src/MD Interpolation"
)

# --- 2D Interpolation Tests ---
add_executable(Linear_test "tests/2DInterpolator/Linear_test.cpp")
target_include_directories(Linear_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Linear_test PRIVATE
    MultiInterpolation
    ThreeDInterpolation
)

add_executable(Cubic_spline_test "tests/2DInterpolator/Cubic_spline_test.cpp")
target_include_directories(Cubic_spline_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Cubic_spline_test PRIVATE
    MultiInterpolation
    ThreeDInterpolation
)

add_executable(Hermite_test "tests/2DInterpolator/Hermite_test.cpp")
target_include_directories(Hermite_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Hermite_test PRIVATE
    MultiInterpolation
    ThreeDInterpolation
)

# --- 3D Interpolation Tests ---
add_executable(Delaune_test "tests/3DInterpolator/Delaune_test.cpp")
target_include_directories(Delaune_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Delaune_test PRIVATE
    MultiInterpolation
    ThreeDInterpolation
)

add_executable(Nearest_Triangle_test "tests/3DInterpolator/Nearest_Triangle_test.cpp")
target_include_directories(Nearest_Triangle_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Nearest_Triangle_test PRIVATE
    MultiInterpolation
    ThreeDInterpolation
)

# --- NN Interpolation Tests ---
add_executable(NN_test "tests/NN/NN_test.cpp")
target_include_directories(NN_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(NN_test PRIVATE
    MultiInterpolation
    NNInterpolation
)

# --- MD Interpolation Tests ---
add_executable(DMD_test "tests/MDInterpolator/Delaune_MD_test.cpp")
target_include_directories(DMD_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(DMD_test PRIVATE
    MultiInterpolation
    MDInterpolation
)

add_executable(NearTrMD_test "tests/MDInterpolator/NearestSimplexInterpolationMD_test.cpp")
target_include_directories(NearTrMD_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(NearTrMD_test PRIVATE
    MultiInterpolation
    MDInterpolation
)

add_executable(Diff_test "tests/MDInterpolator/diff.cpp")
target_include_directories(Diff_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(Diff_test PRIVATE
    MultiInterpolation
    MDInterpolation
)

add_executable(NSNN_test "tests/MDInterpolator/NS+NN.cpp")
target_include_directories(NSNN_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(NSNN_test PRIVATE
    MultiInterpolation
    MDInterpolation
)

# Install all test binaries
install(TARGETS
    Linear_test
    Cubic_spline_test
    Hermite_test
    Delaune_test
    Nearest_Triangle_test
    NN_test
    DMD_test
    NearTrMD_test
    Diff_test
    NSNN_test
  DESTINATION bin
)
cmake_minimum_required(VERSION 3.12)
project(MD-ML)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (MSVC)
    message(FATAL_ERROR "MSVC is not supported, we recommend MinGW-w64 on Windows")
endif ()

set(BOOST_MIN_VERSION "1.70.0") # This is the minimum version of Boost that we have tested with

set(ONLINE_LIB) # List of libraries to link against for online execution
set(FAKE_OFFLINE_LIB) # List of libraries to link against for fake offline execution

include_directories(src)

# Import Eigen, https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package(Eigen3 REQUIRED NO_MODULE)
list(APPEND ONLINE_LIB Eigen3::Eigen)
list(APPEND FAKE_OFFLINE_LIB Eigen3::Eigen)

# Import Boost, we only need the header-only Boost::Asio library,
# so we do not specify components, nor do we add it to the list of libraries
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# For Windows, we need to link against ws2_32 and wsock32 for sockets
if (WIN32)
    list(APPEND ONLINE_LIB ws2_32 wsock32)
endif ()

# Linkage against threads library is explicitly required for Ubuntu 20.04 or earlier
find_package(Threads REQUIRED)
list(APPEND ONLINE_LIB Threads::Threads)

# Create a directory for fake offline data, and pass it to the compiler
set(FAKE_OFFLINE_DIR "${PROJECT_SOURCE_DIR}/fake-offline-data")
add_compile_definitions(FAKE_OFFLINE_DIR="${FAKE_OFFLINE_DIR}")

# Macros for debugging
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(MD_ML_DEBUG)
    add_compile_definitions(MD_ML_DEBUG_ASIO)
endif ()

add_subdirectory(src/networking)

set(SRC_SHARE
        src/share/Spdz2kShare.h
        src/share/Mod2PowN.h
        src/share/IsSpdz2kShare.h
)

set(SRC_FAKE_OFFLINE
        src/fake-offline/FakeParty.h
        src/fake-offline/FakeCircuit.h
        src/fake-offline/FakeGate.h
        src/fake-offline/FakeInputGate.h
        src/fake-offline/FakeAddGate.h
        src/fake-offline/FakeSubtractGate.h
        src/fake-offline/FakeMultiplyGate.h
        src/fake-offline/FakeOutputGate.h
        src/fake-offline/FakeMultiplyTruncGate.h
        src/fake-offline/FakeConv2DGate.h
        src/fake-offline/FakeGtzGate.h
        src/fake-offline/FakeConv2DTruncGate.h
        src/fake-offline/FakeAddConstantGate.h
        src/fake-offline/FakeReLUGate.h
        src/fake-offline/FakeElemMultiplyGate.h
        src/fake-offline/FakeAvgPool2DGate.h
)

set(SRC_UTILS
        src/utils/rand.h
        src/utils/uint128_io.h
        src/utils/linear_algebra.h
        src/utils/print_vector.h
        src/utils/fixed_point.h
        src/utils/tensor.h
)

set(SRC_PROTOCOLS
        src/protocols/Gate.h
        src/protocols/PartyWithFakeOffline.h
        src/protocols/AddGate.h
        src/protocols/InputGate.h
        src/protocols/OutputGate.h
        src/protocols/Circuit.h
        src/protocols/MultiplyGate.h
        src/protocols/SubtractGate.h
        src/protocols/MultiplyTruncGate.h
        src/protocols/Conv2DGate.h
        src/protocols/GtzGate.h
        src/protocols/Conv2DTruncGate.h
        src/protocols/AddConstantGate.h
        src/protocols/ElemMultiplyGate.h
        src/protocols/ReLUGate.h
        src/protocols/AvgPool2DGate.h
)

set(SRC_FILES
        ${SRC_SHARE}
        ${SRC_FAKE_OFFLINE}
        ${SRC_PROTOCOLS}
        ${SRC_UTILS}
)


set(SRC_TIMER
        src/utils/Timer.h
        src/utils/Timer.cpp)
add_library(md-ml-timer ${SRC_TIMER})
list(APPEND ONLINE_LIB md-ml-timer)

add_subdirectory(experiments)

add_executable(dummy dummy.cpp ${SRC_FILES})

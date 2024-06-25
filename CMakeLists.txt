cmake_minimum_required(VERSION 3.25)
project(MD-ML)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)    # We only need the header-only Boost::Asio library

include_directories(src ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})

# Create a directory for fake offline data, and pass it to the compiler
set(FAKE_OFFLINE_DIR "${PROJECT_SOURCE_DIR}/fake-offline-data")
file(MAKE_DIRECTORY ${FAKE_OFFLINE_DIR})
add_compile_definitions(FAKE_OFFLINE_DIR="${FAKE_OFFLINE_DIR}")

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


set(ONLINE_LIB
        Eigen3::Eigen
        md-ml-networking
        md-ml-timer
)

set(FAKE_OFFLINE_LIB
        Eigen3::Eigen
)

add_subdirectory(experiments)

add_executable(dummy dummy.cpp ${SRC_FILES})
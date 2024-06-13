cmake_minimum_required(VERSION 3.25)
project(MD-ML)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
include_directories(src ${EIGEN3_INCLUDE_DIR})

# Create a directory for fake offline data, and pass it to the compiler
set(FAKE_OFFLINE_DIR ${PROJECT_SOURCE_DIR}/fake-offline-data)
file(MAKE_DIRECTORY ${FAKE_OFFLINE_DIR})
add_compile_definitions(FAKE_OFFLINE_DIR="${FAKE_OFFLINE_DIR}")

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
)

set(SRC_UTILS
        src/utils/rand.h
        src/utils/uint128_IO.h
        src/utils/linear_algebra.h
)

set(SRC_FILES
        ${SRC_SHARE}
        ${SRC_FAKE_OFFLINE}
        ${SRC_UTILS}
)

add_executable(test test.cpp ${SRC_FILES})

target_include_directories(test PRIVATE src)
target_link_libraries(test Eigen3::Eigen)
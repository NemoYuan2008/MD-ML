cmake_minimum_required(VERSION 3.25)
project(MD-ML)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

add_executable(test test.cpp src/share/Spdz2kShare.h src/share/Mod2PowN.h
        src/fake-offline/FakeParty.h)
target_include_directories(test PRIVATE src)
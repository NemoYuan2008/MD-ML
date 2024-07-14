# MD-ML

This is the repo of the paper `MD-ML: Super Fast Privacy-Preserving Machine Learning for Malicious Security with a Dishonest Majority`.
We are currently engaged in extensive efforts to refactor our code and to write the documentation.

## Building

### Dependencies

The code has been tested on Windows 11, Ubuntu 22.04, and macOS 14.5. Building the code requires the following dependencies:

- A C++20-compatible compiler
- CMake (3.12 or later)
- The Boost Library (1.70.0 or later)
- The Eigen Library (3.0 or later)

On Ubuntu, you can install all the dependencies via:

```shell
sudo apt install build-essential cmake libboost-system-dev libeigen3-dev
```

On macOS, you can install them using HomeBrew:

```shell
xcode-select --install
brew install cmake boost eigen
```

On Windows, the MSVC compiler is not supported, please use gcc or clang. We recommend using [MinGW-w64](https://www.mingw-w64.org/downloads/#mingw-builds).

### Building

First clone the project and create the build directory:

```shell
git clone https://github.com/NemoYuan2008/MD-ML.git
cd MD-ML
mkdir build
cd build
```

Then configure and build the project:

```shell
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

On Windows with MinGW-w64, you might need to replace the last two commands with

```shell
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

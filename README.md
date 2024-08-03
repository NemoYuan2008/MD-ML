# MD-ML

This is the repo of the paper `MD-ML: Super Fast Privacy-Preserving Machine Learning for Malicious Security with a Dishonest Majority`.
We are currently engaged in extensive efforts to refactor our code and to write the documentation.

## Building

### Dependencies

The code has been tested on Windows 11, Ubuntu 22.04, and macOS 14.5. Building the code requires the following dependencies:

- A C++20-compatible compiler
  - GCC-10 or later, Clang-10 or later, latest version of Apple-Clang.
  - MSVC compiler is not supported, we recommend [MinGW-w64](https://www.mingw-w64.org/downloads/#mingw-builds) on Windows.
- CMake (3.12 or later)
- The Boost Library (1.70.0 or later)
- The Eigen Library (3.0 or later)

On Ubuntu, you can install all the dependencies via:

```shell
sudo apt install build-essential cmake libboost-system-dev libeigen3-dev
```

On macOS, you can install them using [HomeBrew](https://brew.sh):

```shell
xcode-select --install # install the command-line tools
brew install cmake boost eigen
```

On Windows, the MSVC compiler is not supported, please use gcc or clang. We recommend using [MinGW-w64](https://www.mingw-w64.org/downloads/#mingw-builds). The steps for configuration is more complicated. Install [MinGW-w64](https://www.mingw-w64.org/downloads/#mingw-builds) and [CMake](https://cmake.org). Then download the source code of [Boost](https://www.boost.org) and [Eigen](https://eigen.tuxfamily.org). To install Eigen, follow [the installation instructions](https://gitlab.com/libeigen/eigen/-/blob/master/INSTALL?ref_type=heads#L19) "Method 2. Installing using CMake", you may need `-G "MinGW Makefiles"` option when invoking `cmake`. For Boost, just unpack the code, no building is required. Finally, Add the path of Boost and Eigen to environment variable `PATH`.

### Building

First clone the project and create the build directory:

```shell
git clone https://github.com/NemoYuan2008/MD-ML.git
cd MD-ML
mkdir build && cd build
```

On Linux and macOS, configure and build the project with:

```shell
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

On Windows with MinGW-w64, you might need to specify  `-G "MinGW Makefiles"` option when invoking `cmake`:

```shell
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

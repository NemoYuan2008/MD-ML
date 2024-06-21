// By Boshi Yuan

#include "Timer.h"

#include <chrono>
#include <iostream>


namespace md_ml {

void Timer::start() {
    start_ = std::chrono::steady_clock::now();
}

void Timer::stop() {
    stop_ = std::chrono::steady_clock::now();
}

long long Timer::elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_ - start_).count();
}

void Timer::printElapsed() const {
    std::cout << "Elapsed time: " << elapsed() << " ms\n";
}


}

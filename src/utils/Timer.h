// By Boshi Yuan

#ifndef MD_ML_TIMER_H
#define MD_ML_TIMER_H

#include <chrono>
#include <iostream>

namespace md_ml {

class Timer {
public:
    Timer() = default;

    void start();

    void stop();

    [[nodiscard]] long long elapsed() const;

    void printElapsed() const;

    template <typename Func, typename... Args>
    long long benchmark(const Func& f, Args&&... args);

private:
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point stop_;
};


template <typename Func, typename... Args>
long long Timer::benchmark(const Func& f, Args&&... args) {
    start();
    f(std::forward<Args>(args)...);
    stop();
    return elapsed();
}

}


#endif //MD_ML_TIMER_H

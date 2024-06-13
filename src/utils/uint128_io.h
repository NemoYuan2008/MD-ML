// By Boshi Yuan
/// @file
/// Defines IO operators for __uint128_t

#ifndef MD_ML_UINT128_IO_H
#define MD_ML_UINT128_IO_H

#include <iostream>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace md_ml {

inline
std::string Uint128ToString(__uint128_t x) {
    std::string result;
    result.reserve(128);

    if (x == 0) {
        return "0";
    }

    while (x > 0) {
        result += static_cast<char>('0' + (x % 10));
        x /= 10;
    }
    std::ranges::reverse(result);
    result.shrink_to_fit();

    return result;
}


inline
__uint128_t StringToUint128(const std::string& str) {
    __uint128_t result = 0;
    for (char c : str) {
        if (c < '0' || c > '9') {
            throw std::runtime_error("Invalid input: Non-digit characters present.");
        }
        result = result * 10 + (c - '0');
    }
    return result;
}


inline
std::ostream& operator<<(std::ostream& os, __uint128_t x) {
    os << Uint128ToString(x);
    return os;
}


inline
std::istream& operator>>(std::istream& is, __uint128_t& x) {
    std::string str;
    is >> str;
    x = StringToUint128(str);
    return is;
}


} // namespace md_ml
#endif //MD_ML_UINT128_IO_H

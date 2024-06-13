// By Boshi Yuan

#ifndef MD_ML_MOD2POWN_H
#define MD_ML_MOD2POWN_H

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace md_ml {

/// Represent an integer modulo 2^N, N should be less than or equal to 128
template <std::size_t N>
using Mod2PowN_t =
std::conditional_t<N <= 32,
                   uint32_t,
                   std::conditional_t<N <= 64,
                                      uint64_t,
                                      __uint128_t>>;

} // namespace md_ml
#endif //MD_ML_MOD2POWN_H

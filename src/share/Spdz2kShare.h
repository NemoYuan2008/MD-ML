// By Boshi Yuan
/// @file

#ifndef MD_ML_SPDZ2KSHARE_H
#define MD_ML_SPDZ2KSHARE_H

#include <cstddef>
#include "Mod2PowN.h"


namespace md_ml {


template <std::size_t K, std::size_t S>
class Spdz2kShare {
    static_assert(K <= 64, "K should be less than or equal to 64");
    static_assert(S <= 64, "S should be less than or equal to 64");

public:
    // We don't declare any data members here, instead the values are stored in gates
    using KType = Mod2PowN_t<K>; // integer over Z_{2^K}
    using SType = Mod2PowN_t<S>; // integer over Z_{2^S}
    using KSType = Mod2PowN_t<K + S>; // integer over Z_{2^{K+S}}

    using ClearType = KType; // The clear value
    using SemiShrType = KSType; // The share of the values held by the each party (without MAC)
    using KeyShrType = SType; // The share of the MAC key held by the each party
    using GlobalKeyType = KSType; // The global key used for MAC

    constexpr static std::size_t kBits = K;
    constexpr static std::size_t sBits = S;
};

using Spdz2kShare32 = Spdz2kShare<32, 32>;
using Spdz2kShare64 = Spdz2kShare<64, 64>;

} // namespace md_ml

#endif //MD_ML_SPDZ2KSHARE_H

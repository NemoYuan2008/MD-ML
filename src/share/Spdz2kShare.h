//
// Created by yuan on 24-6-7.
//

#ifndef MD_ML_SPDZ2KSHARE_H
#define MD_ML_SPDZ2KSHARE_H


#include "Mod2PowN.h"


template<std::size_t K, std::size_t S>
class Spdz2kShare {
    static_assert(K <= 64, "K should be less than or equal to 64");
    static_assert(S <= 64, "S should be less than or equal to 64");

public:
    // We don't declare any data members here, instead the values are stored in gates
    using KType = Mod2PowN_t<K>;
    using SType = Mod2PowN_t<S>;
    using KSType = Mod2PowN_t<K + S>;

    using ClearType = KType;
    using PartyKeyType = SType;
    using SemiShrType = KSType;     // We call it SemiShr since it's not a full Spdz2kShare (no MAC)
};


using Spdz2kShare32 = Spdz2kShare<32, 32>;
using Spdz2kShare64 = Spdz2kShare<64, 64>;

#endif //MD_ML_SPDZ2KSHARE_H

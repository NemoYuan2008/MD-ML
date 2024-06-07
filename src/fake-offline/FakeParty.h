//
// Created by yuan on 24-6-7.
//

#ifndef MD_ML_FAKEPARTY_H
#define MD_ML_FAKEPARTY_H

#include <array>

template <typename ShrType, int N>
class FakeParty {
public:
    using ClearType = typename ShrType::ClearType;
    using Shares = std::array<ShrType, N>;
};

#endif //MD_ML_FAKEPARTY_H

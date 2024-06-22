// By Boshi Yuan

#ifndef FAKEMULTIPLYTRUNCGATE_H
#define FAKEMULTIPLYTRUNCGATE_H


#include <memory>
#include <vector>
#include <array>
#include <stdexcept>

#include "utils/linear_algebra.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"
#include "fake-offline/FakeMultiplyGate.h"
#include "utils/fixed_point.h"


namespace md_ml {

template <IsSpdz2kShare ShrType, std::size_t N>
class FakeMultiplyTruncGate : public FakeMultiplyGate<ShrType, N> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    FakeMultiplyTruncGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                          const std::shared_ptr<FakeGate<ShrType, N>>& p_input_y);

private:
    void doRunOffline() override;
};


template <IsSpdz2kShare ShrType, std::size_t N>
FakeMultiplyTruncGate<ShrType, N>::
FakeMultiplyTruncGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                      const std::shared_ptr<FakeGate<ShrType, N>>& p_input_y)
    : FakeMultiplyGate<ShrType, N>(p_input_x, p_input_y) {
    // Check and set dimensions
    if (p_input_x->dim_col() != p_input_y->dim_row()) {
        throw std::invalid_argument("The inputs of multiplication gate should have compatible dimensions");
    }
}


template <IsSpdz2kShare ShrType, std::size_t N>
void FakeMultiplyTruncGate<ShrType, N>::doRunOffline() {
    FakeMultiplyGate<ShrType, N>::doRunOffline();

    // The lambda_clear in the base class is actually lambda_prime_clear here
    // So we compute the real lambda_clear in lambda_prime_clear,
    // the real lambda_shr in lambda_prime_shr, and the real lambda_shr_mac in lambda_prime_shr_mac.
    // Then swap the corresponding arrays and vectors.

    std::vector<ClearType> lambda_prime_clear;
    std::array<std::vector<SemiShrType>, N> lambda_prime_shr;
    std::array<std::vector<SemiShrType>, N> lambda_prime_shr_mac;

    lambda_prime_clear = truncateClearVec(this->lambda_clear());
    auto lambda_prime_share_with_mac = this->fake_party().GenerateAllPartiesShares(lambda_prime_clear);
    lambda_prime_shr = std::move(lambda_prime_share_with_mac.value_shares);
    lambda_prime_shr_mac = std::move(lambda_prime_share_with_mac.mac_shares);

    // Write the values to the files
    this->fake_party().WriteSharesToAllParites(lambda_prime_shr);
    this->fake_party().WriteSharesToAllParites(lambda_prime_shr_mac);

    // Swap the vectors
    lambda_prime_clear.swap(this->lambda_clear());
    lambda_prime_shr.swap(this->lambda_shr());
    lambda_prime_shr_mac.swap(this->lambda_shr_mac());
}

} // namespace md_ml


#endif //FAKEMULTIPLYTRUNCGATE_H

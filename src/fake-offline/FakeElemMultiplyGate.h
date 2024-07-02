// By Boshi Yuan

#ifndef FAKEELEMMULTIPLICATIONGATE_H
#define FAKEELEMMULTIPLICATIONGATE_H

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>

#include "utils/linear_algebra.h"
#include "utils/rand.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"

namespace md_ml {


/// @brief Fake offline element-wise multiplication gate (no truncation)
template <IsSpdz2kShare ShrType, std::size_t N>
class FakeElemMultiplyGate : public FakeGate<ShrType, N> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    FakeElemMultiplyGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                         const std::shared_ptr<FakeGate<ShrType, N>>& p_input_y);

protected:
    void doRunOffline() override;
};

template <IsSpdz2kShare ShrType, std::size_t N>
FakeElemMultiplyGate<ShrType, N>::
FakeElemMultiplyGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                     const std::shared_ptr<FakeGate<ShrType, N>>& p_input_y)
    : FakeGate<ShrType, N>(p_input_x, p_input_y) {
    // Check and set dimensions
    if (p_input_x->dim_row() != p_input_y->dim_row()
        || p_input_x->dim_col() != p_input_y->dim_col()) {
        throw std::invalid_argument("The inputs of element-wise multiplication gate should have compatible dimensions");
    }
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeElemMultiplyGate<ShrType, N>::doRunOffline() {
    auto size = this->dim_row() * this->dim_col();

    // $\lambda_z$ = rand()
    this->lambda_clear().resize(size);
    std::ranges::generate(this->lambda_clear(), getRand<ClearType>);

    auto lambda_shares_and_macs = this->fake_party().GenerateAllPartiesShares(this->lambda_clear());
    this->lambda_shr() = std::move(lambda_shares_and_macs.value_shares);
    this->lambda_shr_mac() = std::move(lambda_shares_and_macs.mac_shares);

    // Generate the element-wise multiplication triples
    std::vector<ClearType> a_clear(size);
    std::vector<ClearType> b_clear(size);
    std::vector<ClearType> c_clear;

    std::ranges::generate(a_clear, getRand<ClearType>);
    std::ranges::generate(b_clear, getRand<ClearType>);
    c_clear = matrixElemMultiply(a_clear, b_clear);

    auto a_share_with_mac = this->fake_party().GenerateAllPartiesShares(a_clear);
    auto b_share_with_mac = this->fake_party().GenerateAllPartiesShares(b_clear);
    auto c_share_with_mac = this->fake_party().GenerateAllPartiesShares(c_clear);

    // $\delta_x = a - \lambda_x$, $\delta_y = b - \lambda_y$
    auto delta_x_clear = matrixSubtract(a_clear, this->input_x()->lambda_clear());
    auto delta_y_clear = matrixSubtract(b_clear, this->input_y()->lambda_clear());

    // Wrtie all data to files
    this->fake_party().WriteSharesToAllParites(a_share_with_mac.value_shares);
    this->fake_party().WriteSharesToAllParites(a_share_with_mac.mac_shares);
    this->fake_party().WriteSharesToAllParites(b_share_with_mac.value_shares);
    this->fake_party().WriteSharesToAllParites(b_share_with_mac.mac_shares);
    this->fake_party().WriteSharesToAllParites(c_share_with_mac.value_shares);
    this->fake_party().WriteSharesToAllParites(c_share_with_mac.mac_shares);
    this->fake_party().WriteSharesToAllParites(this->lambda_shr());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr_mac());

    this->fake_party().WriteClearToAllParties(delta_x_clear);
    this->fake_party().WriteClearToAllParties(delta_y_clear);
}

} // namespace md_ml

#endif //FAKEELEMMULTIPLICATIONGATE_H

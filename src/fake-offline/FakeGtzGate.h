// By Boshi Yuan

#ifndef FAKEGTZGATE_H
#define FAKEGTZGATE_H

#include <memory>
#include <array>
#include <algorithm>

#include "utils/rand.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"


namespace md_ml {

template <IsSpdz2kShare ShrType, std::size_t N>
class FakeGtzGate : public FakeGate<ShrType, N> {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    explicit FakeGtzGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x);

private:
    static std::array<ClearType, N> generateBooleanShares(ClearType x);
    void doRunOffline() override;
};

template <IsSpdz2kShare ShrType, std::size_t N>
FakeGtzGate<ShrType, N>::
FakeGtzGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x)
    : FakeGate<ShrType, N>(p_input_x, nullptr) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::array<typename FakeGtzGate<ShrType, N>::ClearType, N>
FakeGtzGate<ShrType, N>::generateBooleanShares(ClearType x) {
    std::array<ClearType, N> ret;
    for (int i = 0; i < N - 1; ++i) {
        ret[i] = getRand<ClearType>();
        x ^= ret[i];
    }
    ret.back() = x;
    return ret;
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeGtzGate<ShrType, N>::doRunOffline() {
    auto size = this->dim_row() * this->dim_col();

    // $[\lambda]$-values are uniformly random
    this->lambda_clear().resize(size);
    std::ranges::generate(this->lambda_clear(), getRand<ClearType>);

    // Generate the shares
    auto shares_and_macs = this->fake_party().GenerateAllPartiesShares(this->lambda_clear());
    this->lambda_shr() = std::move(shares_and_macs.value_shares);
    this->lambda_shr_mac() = std::move(shares_and_macs.mac_shares);

    // Write the values to the output files
    this->fake_party().WriteSharesToAllParites(this->lambda_shr());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr_mac());

    // Boolean shares of lambda_x
    // TODO: clean up the code, extract the boolean share generation to a function
    for (std::size_t vec_idx = 0; vec_idx < size; ++vec_idx) {
        auto shares_i = generateBooleanShares(this->lambda_clear()[vec_idx]);
        for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
            this->fake_party().ithPartyFile(party_idx) << shares_i[party_idx] << '\n';
        }
    }
}


} // namespace md_ml

#endif //FAKEGTZGATE_H

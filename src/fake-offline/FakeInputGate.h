// By Boshi Yuan

#ifndef MD_ML_FAKEINPUTGATE_H
#define MD_ML_FAKEINPUTGATE_H

#include <memory>
#include <algorithm>

#include "utils/rand.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"


namespace md_ml {


template <IsSpdz2kShare ShrType, std::size_t N>
class FakeInputGate : public FakeGate<ShrType, N> {
public:
    using ClearType = typename ShrType::ClearType;

    FakeInputGate(FakeParty<ShrType, N>& p_fake_party,
                  std::size_t p_dim_row, std::size_t p_dim_col,
                  std::size_t p_owner_id);

private:
    void doRunOffline() override;

    size_t owner_id_;
};


template <IsSpdz2kShare ShrType, std::size_t N>
FakeInputGate<ShrType, N>::FakeInputGate(FakeParty<ShrType, N>& p_fake_party,
                                         std::size_t p_dim_row, std::size_t p_dim_col,
                                         std::size_t p_owner_id)
    : FakeGate<ShrType, N>(p_fake_party, p_dim_row, p_dim_col), owner_id_(p_owner_id) {}


template <IsSpdz2kShare ShrType, std::size_t N>
void FakeInputGate<ShrType, N>::doRunOffline() {
    auto size = this->dim_row() * this->dim_col();

    // $[\lambda]$-values are uniformly random
    this->lambda_clear().resize(size);
    std::ranges::generate(this->lambda_clear(), getRand<ClearType>);

    // Generate the shares
    auto shares_and_macs = this->fake_party().GenerateAllPartiesShares(this->lambda_clear());
    this->lambda_shr() = std::move(shares_and_macs.value_shares);
    this->lambda_shr_mac() = std::move(shares_and_macs.mac_shares);

    // Write the values to the output files
    this->fake_party().WriteClearToIthParty(this->lambda_clear(), owner_id_); // Owner should know lambda_clear
    // this->fake_party().WriteSharesToAllParites(this->lambda_shr(), this->lambda_shr_mac());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr_mac());
}


} // namespace md_ml

#endif //MD_ML_FAKEINPUTGATE_H

// By Boshi Yuan

#ifndef FAKEINPUTGATE_H
#define FAKEINPUTGATE_H

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

    // this->lambda_clear().resize(size);
    // std::ranges::for_each(this->lambda_shr(), [size](auto& vec) { vec.resize(size); });
    // std::ranges::for_each(this->lambda_shr_mac(), [size](auto& vec) { vec.resize(size); });

    // // Generate the shared lambda values for all parties
    // for (std::size_t vec_idx = 0; vec_idx < size; ++vec_idx) {
    //     auto lambda_clear_i = getRand<ClearType>(); // $\lambda$-values are uniformly random
    //     this->lambda_clear()[vec_idx] = lambda_clear_i;
    //
    //     auto lambda_shares_i = this->fake_party().GenerateAllPartiesShares(lambda_clear_i);
    //     for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
    //         this->lambda_shr()[party_idx][vec_idx] = lambda_shares_i.value_shares[party_idx];
    //         this->lambda_shr_mac()[party_idx][vec_idx] = lambda_shares_i.mac_shares[party_idx];
    //     }
    // }

    // // Write the shared lambda values to the output files
    // for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
    //     auto& output_file = this->fake_party().ithPartyFile(party_idx);
    //
    //     for (std::size_t vec_idx = 0; vec_idx < size; ++vec_idx) {
    //         if (party_idx == owner_id_) {
    //             // The owner of the input should know lambda_clear
    //             output_file << this->lambda_clear()[vec_idx] << ' ';
    //         }
    //
    //         output_file << this->lambda_shr()[party_idx][vec_idx] << ' '
    //             << this->lambda_shr_mac()[party_idx][vec_idx] << '\n';
    //     }
    // }

    // $[\lambda]$-values are uniformly random
    this->lambda_clear().resize(size);
    std::ranges::generate(this->lambda_clear(), getRand<ClearType>);

    // Generate the shares
    auto shares_and_macs = this->fake_party().GenerateAllPartiesShares(this->lambda_clear());
    this->lambda_shr() = std::move(shares_and_macs.value_shares);
    this->lambda_shr_mac() = std::move(shares_and_macs.mac_shares);

    // Write the values to the output files
    this->fake_party().WriteClearToIthParty(this->lambda_clear(), owner_id_); // Owner should know lambda_clear
    this->fake_party().WriteSharesToAllParites(this->lambda_shr(), this->lambda_shr_mac());
}


} // namespace md_ml

#endif //FAKEINPUTGATE_H

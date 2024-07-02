// By Boshi Yuan

#ifndef FAKEAVGPOOL2DGATE_H
#define FAKEAVGPOOL2DGATE_H

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>

#include "utils/linear_algebra.h"
#include "utils/tensor.h"
#include "utils/fixed_point.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"


namespace md_ml {

template <IsSpdz2kShare ShrType, std::size_t N>
class FakeAvgPool2DGate : public FakeGate<ShrType, N> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    FakeAvgPool2DGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                      const MaxPoolOp& op);

private:
    void doRunOffline() override;

    MaxPoolOp maxPoolOp;
};

template <IsSpdz2kShare ShrType, std::size_t N>
FakeAvgPool2DGate<ShrType, N>::
FakeAvgPool2DGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                  const MaxPoolOp& op)
    : FakeGate<ShrType, N>(p_input_x, nullptr), maxPoolOp(op) {
    this->set_dim_row(op.compute_output_size());
    this->set_dim_col(1);
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeAvgPool2DGate<ShrType, N>::doRunOffline() {
    auto size = maxPoolOp.compute_output_size();

    std::vector<ClearType> lambda_pre_trunc_clear(size);
    std::ranges::generate(lambda_pre_trunc_clear, getRand<ClearType>);

    this->lambda_clear() = truncateClearVec(lambda_pre_trunc_clear);
    auto shares_and_macs = this->fake_party().GenerateAllPartiesShares(this->lambda_clear());
    this->lambda_shr() = std::move(shares_and_macs.value_shares);
    this->lambda_shr_mac() = std::move(shares_and_macs.mac_shares);

    std::array<std::vector<SemiShrType>, N> lambda_pre_trunc_shr, lambda_pre_trunc_shr_mac;
    shares_and_macs = this->fake_party().GenerateAllPartiesShares(lambda_pre_trunc_clear);
    lambda_pre_trunc_shr = std::move(shares_and_macs.value_shares);
    lambda_pre_trunc_shr_mac = std::move(shares_and_macs.mac_shares);

    this->fake_party().WriteSharesToAllParites(this->lambda_shr());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr_mac());
    this->fake_party().WriteSharesToAllParites(lambda_pre_trunc_shr);
    this->fake_party().WriteSharesToAllParites(lambda_pre_trunc_shr_mac);
}

} // namespace md_ml


#endif //FAKEAVGPOOL2DGATE_H

// By Boshi Yuan

#ifndef MD_ML_FAKESUBTRACTGATE_H
#define MD_ML_FAKESUBTRACTGATE_H

#include <memory>
#include <stdexcept>

#include "utils/linear_algebra.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"


namespace md_ml {


template <IsSpdz2kShare ShrType, std::size_t N>
class FakeSubtractGate : public FakeGate<ShrType, N> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    FakeSubtractGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                     const std::shared_ptr<FakeGate<ShrType, N>>& p_input_y);

private:
    void doRunOffline() override;
};

template <IsSpdz2kShare ShrType, std::size_t N>
FakeSubtractGate<ShrType, N>::
FakeSubtractGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                 const std::shared_ptr<FakeGate<ShrType, N>>& p_input_y)
    : FakeGate<ShrType, N>(p_input_x, p_input_y) {
    if (p_input_x->dim_row() != p_input_y->dim_row() ||
        p_input_x->dim_col() != p_input_y->dim_col()) {
        throw std::invalid_argument("The inputs of subtraction gate should have the same dimensions");
    }
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeSubtractGate<ShrType, N>::doRunOffline() {
    auto size = this->dim_row() * this->dim_col();

    // $[\lambda_z] = [\lambda_x] - [\lambda_y]$
    for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
        this->lambda_shr()[party_idx] = matrixSubtract(this->input_x()->lambda_shr()[party_idx],
                                                       this->input_y()->lambda_shr()[party_idx]);
        this->lambda_shr_mac()[party_idx] = matrixSubtract(this->input_x()->lambda_shr_mac()[party_idx],
                                                           this->input_y()->lambda_shr_mac()[party_idx]);
    }

    // this->fake_party().WriteSharesToAllParites(this->lambda_shr(), this->lambda_shr_mac());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr());
    this->fake_party().WriteSharesToAllParites(this->lambda_shr_mac());
}


} // namespace md_ml


#endif //MD_ML_FAKESUBTRACTGATE_H

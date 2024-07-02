// By Boshi Yuan

#ifndef FAKEADDCONSTANTGATE_H
#define FAKEADDCONSTANTGATE_H

#include <memory>
#include <stdexcept>

#include "utils/linear_algebra.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"


namespace md_ml {

template <IsSpdz2kShare ShrType, std::size_t N>
class FakeAddConstantGate : public FakeGate<ShrType, N> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    FakeAddConstantGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                        const ClearType& constant);

private:
    void doRunOffline() override;

    ClearType constant_;
};

template <IsSpdz2kShare ShrType, std::size_t N>
FakeAddConstantGate<ShrType, N>::
FakeAddConstantGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x,
                    const ClearType& constant)
    : FakeGate<ShrType, N>(p_input_x, nullptr), constant_(constant) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeAddConstantGate<ShrType, N>::doRunOffline() {
    this->lambda_clear() = this->input_x()->lambda_clear();
    this->lambda_shr() = this->input_x()->lambda_shr();
    this->lambda_shr_mac() = this->input_x()->lambda_shr_mac();
}

} // namespace md_ml

#endif //FAKEADDCONSTANTGATE_H

// By Boshi Yuan

#ifndef ADDCONSTANTGATE_H
#define ADDCONSTANTGATE_H

#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class AddConstantGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    AddConstantGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                    const ClearType& constant);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    ClearType constant_;
};

template <IsSpdz2kShare ShrType>
AddConstantGate<ShrType>::
AddConstantGate(const std::shared_ptr<Gate<ShrType>>& p_input_x, const ClearType& constant)
    : Gate<ShrType>(p_input_x, nullptr), constant_(constant) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType>
void AddConstantGate<ShrType>::doReadOfflineFromFile() {
    this->lambda_shr() = this->input_x()->lambda_shr();
    this->lambda_shr_mac() = this->input_x()->lambda_shr_mac();
}

template <IsSpdz2kShare ShrType>
void AddConstantGate<ShrType>::doRunOnline() {
    this->Delta_clear() = matrixAddConstant(this->input_x()->Delta_clear(), constant_);
}

} // md_ml

#endif //ADDCONSTANTGATE_H

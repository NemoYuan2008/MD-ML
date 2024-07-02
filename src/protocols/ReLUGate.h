// By Boshi Yuan

#ifndef RELUGATE_H
#define RELUGATE_H

#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class Circuit;


template <IsSpdz2kShare ShrType>
class ReLUGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    explicit ReLUGate(const std::shared_ptr<Gate<ShrType>>& input_x);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    Circuit<ShrType> circuit_;
};

template <IsSpdz2kShare ShrType>
ReLUGate<ShrType>::
ReLUGate(const std::shared_ptr<Gate<ShrType>>& input_x)
    : Gate<ShrType>(input_x, nullptr), circuit_(input_x->party()) {
    this->set_dim_row(input_x->dim_row());
    this->set_dim_col(input_x->dim_col());

    auto b = circuit_.gtz(input_x);
    auto z = circuit_.elementMultiply(this->input_x(), b);
    circuit_.addEndpoint(z);
}

template <IsSpdz2kShare ShrType>
void ReLUGate<ShrType>::doReadOfflineFromFile() {
    circuit_.readOfflineFromFile();
    this->lambda_shr() = this->circuit_.endpoints()[0]->lambda_shr();
}

template <IsSpdz2kShare ShrType>
void ReLUGate<ShrType>::doRunOnline() {
    circuit_.runOnline();
    this->Delta_clear() = this->circuit_.endpoints()[0]->Delta_clear();
}

} // namespace md_ml

#endif //RELUGATE_H

// By Boshi Yuan

#ifndef FAKERELUGATE_H
#define FAKERELUGATE_H

#include <memory>
#include <stdexcept>

#include "utils/linear_algebra.h"
#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeGate.h"
#include "fake-offline/FakeCircuit.h"

namespace md_ml {

template <IsSpdz2kShare ShrType, std::size_t N>
class FakeReLUGate : public FakeGate<ShrType, N> {
public:
    explicit FakeReLUGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x);

private:
    FakeCircuit<ShrType, N> circuit_;
};

template <IsSpdz2kShare ShrType, std::size_t N>
FakeReLUGate<ShrType, N>::
FakeReLUGate(const std::shared_ptr<FakeGate<ShrType, N>>& p_input_x)
    : FakeGate<ShrType, N>(p_input_x, nullptr), circuit_(this->fake_party()) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());

    auto b = this->circuit_.gtz(this->input_x());
    // TODO: Implement elementMultiply first
}

} // md_ml

#endif //FAKERELUGATE_H

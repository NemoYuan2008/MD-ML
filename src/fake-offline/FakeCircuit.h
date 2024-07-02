// By Boshi Yuan

#ifndef MD_ML_FAKECIRCUIT_H
#define MD_ML_FAKECIRCUIT_H


#include <memory>
#include <vector>
#include <cstddef>

#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeParty.h"
#include "fake-offline/FakeGate.h"
#include "fake-offline/FakeInputGate.h"
#include "fake-offline/FakeAddGate.h"
#include "fake-offline/FakeSubtractGate.h"
#include "fake-offline/FakeAddConstantGate.h"
#include "fake-offline/FakeMultiplyGate.h"
#include "fake-offline/FakeOutputGate.h"
#include "fake-offline/FakeMultiplyTruncGate.h"
#include "fake-offline/FakeConv2DGate.h"
#include "fake-offline/FakeConv2DTruncGate.h"
#include "fake-offline/FakeGtzGate.h"


namespace md_ml {


/// @brief A fake circuit that uses the fake offline protocol to compute the result
/// @tparam ShrType
/// @tparam N
template <IsSpdz2kShare ShrType, std::size_t N>
class FakeCircuit {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    explicit FakeCircuit(FakeParty<ShrType, N>& p_fake_party) : fake_party_(p_fake_party) {}

    void runOffline();

    void addEndpoint(const std::shared_ptr<FakeGate<ShrType, N>>& gate);

    std::shared_ptr<FakeInputGate<ShrType, N>>
    input(std::size_t owner_id, std::size_t dim_row, std::size_t dim_col);

    std::shared_ptr<FakeAddGate<ShrType, N>>
    add(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
        const std::shared_ptr<FakeGate<ShrType, N>>& input_y);

    std::shared_ptr<FakeSubtractGate<ShrType, N>>
    subtract(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
             const std::shared_ptr<FakeGate<ShrType, N>>& input_y);

    std::shared_ptr<FakeAddConstantGate<ShrType, N>>
    addConstant(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
                ClearType constant);

    std::shared_ptr<FakeMultiplyGate<ShrType, N>>
    multiply(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
             const std::shared_ptr<FakeGate<ShrType, N>>& input_y);

    std::shared_ptr<FakeOutputGate<ShrType, N>>
    output(const std::shared_ptr<FakeGate<ShrType, N>>& input_x);

    std::shared_ptr<FakeMultiplyTruncGate<ShrType, N>>
    multiplyTrunc(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
                  const std::shared_ptr<FakeGate<ShrType, N>>& input_y);

    std::shared_ptr<FakeConv2DGate<ShrType, N>>
    conv2D(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
           const std::shared_ptr<FakeGate<ShrType, N>>& input_y,
           const Conv2DOp& op);

    std::shared_ptr<FakeConv2DTruncGate<ShrType, N>>
    conv2DTrunc(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
                const std::shared_ptr<FakeGate<ShrType, N>>& input_y,
                const Conv2DOp& op);

    std::shared_ptr<FakeGtzGate<ShrType, N>>
    gtz(const std::shared_ptr<FakeGate<ShrType, N>>& input_x);

private:
    FakeParty<ShrType, N>& fake_party_;
    std::vector<std::shared_ptr<FakeGate<ShrType, N>>> gates_;
    std::vector<std::shared_ptr<FakeGate<ShrType, N>>> endpoints_;
};


template <IsSpdz2kShare ShrType, std::size_t N>
void FakeCircuit<ShrType, N>::
runOffline() {
    for (const auto& gatePtr : endpoints_) {
        gatePtr->runOffline();
    }
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeCircuit<ShrType, N>::
addEndpoint(const std::shared_ptr<FakeGate<ShrType, N>>& gate) {
    endpoints_.push_back(gate);
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeInputGate<ShrType, N>> FakeCircuit<ShrType, N>::
input(std::size_t owner_id, std::size_t dim_row, std::size_t dim_col) {
    auto gate = std::make_shared<FakeInputGate<ShrType, N>>(fake_party_, dim_row, dim_col, owner_id);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeAddGate<ShrType, N>> FakeCircuit<ShrType, N>::
add(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
    const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
    auto gate = std::make_shared<FakeAddGate<ShrType, N>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeSubtractGate<ShrType, N>> FakeCircuit<ShrType, N>::
subtract(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
         const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
    auto gate = std::make_shared<FakeSubtractGate<ShrType, N>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeAddConstantGate<ShrType, N>> FakeCircuit<ShrType, N>::
addConstant(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
            ClearType constant) {
    auto gate = std::make_shared<FakeAddConstantGate<ShrType, N>>(input_x, constant);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeMultiplyGate<ShrType, N>> FakeCircuit<ShrType, N>::
multiply(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
         const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
    auto gate = std::make_shared<FakeMultiplyGate<ShrType, N>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeOutputGate<ShrType, N>> FakeCircuit<ShrType, N>::
output(const std::shared_ptr<FakeGate<ShrType, N>>& input_x) {
    auto gate = std::make_shared<FakeOutputGate<ShrType, N>>(input_x);
    gates_.push_back(gate);
    // TODO: shall we add the output gate to the endpoints?
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeMultiplyTruncGate<ShrType, N>> FakeCircuit<ShrType, N>::
multiplyTrunc(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
              const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
    auto gate = std::make_shared<FakeMultiplyTruncGate<ShrType, N>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeConv2DGate<ShrType, N>> FakeCircuit<ShrType, N>::
conv2D(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
       const std::shared_ptr<FakeGate<ShrType, N>>& input_y,
       const Conv2DOp& op) {
    auto gate = std::make_shared<FakeConv2DGate<ShrType, N>>(input_x, input_y, op);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeConv2DTruncGate<ShrType, N>> FakeCircuit<ShrType, N>::
conv2DTrunc(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
            const std::shared_ptr<FakeGate<ShrType, N>>& input_y,
            const Conv2DOp& op) {
    auto gate = std::make_shared<FakeConv2DTruncGate<ShrType, N>>(input_x, input_y, op);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType, std::size_t N>
std::shared_ptr<FakeGtzGate<ShrType, N>> FakeCircuit<ShrType, N>::
gtz(const std::shared_ptr<FakeGate<ShrType, N>>& input_x) {
    auto gate = std::make_shared<FakeGtzGate<ShrType, N>>(input_x);
    gates_.push_back(gate);
    return gate;
}

} // namespace md_ml

#endif //MD_ML_FAKECIRCUIT_H

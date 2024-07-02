// By Boshi Yuan

#ifndef MD_ML_CIRCUIT_H
#define MD_ML_CIRCUIT_H


#include <memory>
#include <vector>

#include "utils/Timer.h"
#include "share/IsSpdz2kShare.h"
#include "protocols/PartyWithFakeOffline.h"
#include "protocols/Gate.h"
#include "protocols/InputGate.h"
#include "protocols/AddGate.h"
#include "protocols/SubtractGate.h"
#include "protocols/MultiplyGate.h"
#include "protocols/OutputGate.h"
#include "protocols/MultiplyTruncGate.h"
#include "protocols/ElemMultiplyGate.h"
#include "protocols/Conv2DGate.h"
#include "protocols/Conv2DTruncGate.h"
#include "protocols/GtzGate.h"
#include "protocols/ReLUGate.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class Circuit {
public:
    explicit Circuit(PartyWithFakeOffline<ShrType>& party) : party_(party) {}

    void addEndPoint(const std::shared_ptr<Gate<ShrType>>& gate);
    void runOffline();
    void readOfflineFromFile();
    void runOnline();
    void runOnlineWithBenckmark();
    void printStats();

    std::shared_ptr<InputGate<ShrType>>
    input(std::size_t owner_id, std::size_t dim_row, std::size_t dim_col);

    std::shared_ptr<AddGate<ShrType>>
    add(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);

    std::shared_ptr<SubtractGate<ShrType>>
    subtract(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);

    std::shared_ptr<MultiplyGate<ShrType>>
    multiply(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);

    std::shared_ptr<OutputGate<ShrType>>
    output(const std::shared_ptr<Gate<ShrType>>& input);

    std::shared_ptr<MultiplyTruncGate<ShrType>>
    multiplyTrunc(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);

    std::shared_ptr<ElemMultiplyGate<ShrType>>
    elementMultiply(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);

    std::shared_ptr<Conv2DGate<ShrType>>
    conv2D(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y,
           const Conv2DOp& op);

    std::shared_ptr<Conv2DTruncGate<ShrType>>
    conv2DTrunc(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y,
                const Conv2DOp& op);

    std::shared_ptr<GtzGate<ShrType>>
    gtz(const std::shared_ptr<Gate<ShrType>>& input_x);

    std::shared_ptr<ReLUGate<ShrType>>
    relu(const std::shared_ptr<Gate<ShrType>>& input_x);

    [[nodiscard]] auto& endpoints() { return endpoints_; }

private:
    PartyWithFakeOffline<ShrType>& party_;
    std::vector<std::shared_ptr<Gate<ShrType>>> gates_;
    std::vector<std::shared_ptr<Gate<ShrType>>> endpoints_;
    Timer timer_;
};


template <IsSpdz2kShare ShrType>
void Circuit<ShrType>::addEndPoint(const std::shared_ptr<Gate<ShrType>>& gate) {
    endpoints_.push_back(gate);
}

template <IsSpdz2kShare ShrType>
void Circuit<ShrType>::runOffline() {
    for (const auto& gate : endpoints_) {
        gate->runOffline();
    }
}

template <IsSpdz2kShare ShrType>
void Circuit<ShrType>::readOfflineFromFile() {
    for (const auto& gate : endpoints_) {
        gate->readOfflineFromFile();
    }
}

template <IsSpdz2kShare ShrType>
void Circuit<ShrType>::runOnline() {
    for (const auto& gate : endpoints_) {
        gate->RunOnline();
    }
}

template <IsSpdz2kShare ShrType>
void Circuit<ShrType>::runOnlineWithBenckmark() {
    timer_.start();
    runOnline();
    timer_.stop();
}

template <IsSpdz2kShare ShrType>
void Circuit<ShrType>::printStats() {
    std::cout
        << "Spent " << timer_.elapsed() << " ms\n"
        << "Sent " << party_.bytes_sent() << " bytes\n";
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<InputGate<ShrType>> Circuit<ShrType>::
input(std::size_t owner_id, std::size_t dim_row, std::size_t dim_col) {
    auto gate = std::make_shared<InputGate<ShrType>>(party_, dim_row, dim_col, owner_id);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<AddGate<ShrType>> Circuit<ShrType>::
add(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<AddGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<SubtractGate<ShrType>> Circuit<ShrType>::
subtract(const std::shared_ptr<Gate<ShrType>>& input_x,
         const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<SubtractGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<MultiplyGate<ShrType>> Circuit<ShrType>::
multiply(const std::shared_ptr<Gate<ShrType>>& input_x,
         const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<MultiplyGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<OutputGate<ShrType>> Circuit<ShrType>::
output(const std::shared_ptr<Gate<ShrType>>& input) {
    auto gate = std::make_shared<OutputGate<ShrType>>(input);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<MultiplyTruncGate<ShrType>> Circuit<ShrType>::
multiplyTrunc(const std::shared_ptr<Gate<ShrType>>& input_x,
              const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<MultiplyTruncGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<ElemMultiplyGate<ShrType>> Circuit<ShrType>::
elementMultiply(const std::shared_ptr<Gate<ShrType>>& input_x,
                const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<ElemMultiplyGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<Conv2DGate<ShrType>> Circuit<ShrType>::
conv2D(const std::shared_ptr<Gate<ShrType>>& input_x,
       const std::shared_ptr<Gate<ShrType>>& input_y,
       const Conv2DOp& op) {
    auto gate = std::make_shared<Conv2DGate<ShrType>>(input_x, input_y, op);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<Conv2DTruncGate<ShrType>> Circuit<ShrType>::
conv2DTrunc(const std::shared_ptr<Gate<ShrType>>& input_x,
            const std::shared_ptr<Gate<ShrType>>& input_y,
            const Conv2DOp& op) {
    auto gate = std::make_shared<Conv2DTruncGate<ShrType>>(input_x, input_y, op);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<GtzGate<ShrType>> Circuit<ShrType>::
gtz(const std::shared_ptr<Gate<ShrType>>& input_x) {
    auto gate = std::make_shared<GtzGate<ShrType>>(input_x);
    gates_.push_back(gate);
    return gate;
}

template <IsSpdz2kShare ShrType>
std::shared_ptr<ReLUGate<ShrType>> Circuit<ShrType>::
relu(const std::shared_ptr<Gate<ShrType>>& input_x) {
    auto gate = std::make_shared<ReLUGate<ShrType>>(input_x);
    gates_.push_back(gate);
    return gate;
}

} // md_ml

#endif //MD_ML_CIRCUIT_H

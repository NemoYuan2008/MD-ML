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

namespace md_ml {

template <typename ShrType>
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

private:
    PartyWithFakeOffline<ShrType>& party_;
    std::vector<std::shared_ptr<Gate<ShrType>>> gates_;
    std::vector<std::shared_ptr<Gate<ShrType>>> endpoints_;
    Timer timer_;
};


template <typename ShrType>
void Circuit<ShrType>::addEndPoint(const std::shared_ptr<Gate<ShrType>>& gate) {
    endpoints_.push_back(gate);
}

template <typename ShrType>
void Circuit<ShrType>::runOffline() {
    for (const auto& gate : endpoints_) {
        gate->runOffline();
    }
}

template <typename ShrType>
void Circuit<ShrType>::readOfflineFromFile() {
    for (const auto& gate : endpoints_) {
        gate->readOfflineFromFile();
    }
}

template <typename ShrType>
void Circuit<ShrType>::runOnline() {
    for (const auto& gate : endpoints_) {
        gate->RunOnline();
    }
}

template <typename ShrType>
void Circuit<ShrType>::runOnlineWithBenckmark() {
    timer_.start();
    runOnline();
    timer_.stop();
}

template <typename ShrType>
void Circuit<ShrType>::printStats() {
    std::cout
        << "Spent " << timer_.elapsed() << " ms\n"
        << "Sent " << party_.bytes_sent() << " bytes\n";
}

template <typename ShrType>
std::shared_ptr<InputGate<ShrType>> Circuit<ShrType>::
input(std::size_t owner_id, std::size_t dim_row, std::size_t dim_col) {
    auto gate = std::make_shared<InputGate<ShrType>>(party_, dim_row, dim_col, owner_id);
    gates_.push_back(gate);
    return gate;
}

template <typename ShrType>
std::shared_ptr<AddGate<ShrType>> Circuit<ShrType>::
add(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<AddGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <typename ShrType>
std::shared_ptr<SubtractGate<ShrType>> Circuit<ShrType>::
subtract(const std::shared_ptr<Gate<ShrType>>& input_x,
         const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<SubtractGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <typename ShrType>
std::shared_ptr<MultiplyGate<ShrType>> Circuit<ShrType>::
multiply(const std::shared_ptr<Gate<ShrType>>& input_x,
         const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<MultiplyGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

template <typename ShrType>
std::shared_ptr<OutputGate<ShrType>> Circuit<ShrType>::
output(const std::shared_ptr<Gate<ShrType>>& input) {
    auto gate = std::make_shared<OutputGate<ShrType>>(input);
    gates_.push_back(gate);
    return gate;
}

template <typename ShrType>
std::shared_ptr<MultiplyTruncGate<ShrType>> Circuit<ShrType>::
multiplyTrunc(const std::shared_ptr<Gate<ShrType>>& input_x,
              const std::shared_ptr<Gate<ShrType>>& input_y) {
    auto gate = std::make_shared<MultiplyTruncGate<ShrType>>(input_x, input_y);
    gates_.push_back(gate);
    return gate;
}

} // md_ml

#endif //MD_ML_CIRCUIT_H

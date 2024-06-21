// By Boshi Yuan

#ifndef MD_ML_CIRCUIT_H
#define MD_ML_CIRCUIT_H


#include <memory>
#include <vector>

#include "share/IsSpdz2kShare.h"
#include "protocols/PartyWithFakeOffline.h"
#include "protocols/Gate.h"
#include "protocols/InputGate.h"
#include "protocols/AddGate.h"
#include "protocols/SubtractGate.h"
#include "protocols/MultiplyGate.h"
#include "protocols/OutputGate.h"

namespace md_ml {

template <typename ShrType>
class Circuit {
public:
    explicit Circuit(PartyWithFakeOffline<ShrType>& party) : party_(party) {}

    void addEndPoint(const std::shared_ptr<Gate<ShrType>>& gate);

    std::shared_ptr<InputGate<ShrType>>
    input(std::size_t owner_id, std::size_t dim_row, std::size_t dim_col);

    std::shared_ptr<AddGate<ShrType>>
    add(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);

    std::shared_ptr<SubtractGate<ShrType>>
    subtract(const std::shared_ptr<Gate<ShrType>>& input_x, const std::shared_ptr<Gate<ShrType>>& input_y);



    std::shared_ptr<OutputGate<ShrType>>
    output(const std::shared_ptr<Gate<ShrType>>& input);

private:
    PartyWithFakeOffline<ShrType>& party_;
    std::vector<std::shared_ptr<Gate<ShrType>>> gates_;
    std::vector<std::shared_ptr<Gate<ShrType>>> endpoints_;
};


template <typename ShrType>
void Circuit<ShrType>::addEndPoint(const std::shared_ptr<Gate<ShrType>>& gate) {
    endpoints_.push_back(gate);
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
std::shared_ptr<OutputGate<ShrType>> Circuit<ShrType>::
output(const std::shared_ptr<Gate<ShrType>>& input) {
    auto gate = std::make_shared<OutputGate<ShrType>>(input);
    gates_.push_back(gate);
    return gate;
}

} // md_ml

#endif //MD_ML_CIRCUIT_H
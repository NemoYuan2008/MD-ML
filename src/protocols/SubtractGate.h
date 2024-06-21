// By Boshi Yuan

#ifndef SUBTRACTGATE_H
#define SUBTRACTGATE_H

#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class SubtractGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    SubtractGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                 const std::shared_ptr<Gate<ShrType>>& p_input_y);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;
};


template <IsSpdz2kShare ShrType>
SubtractGate<ShrType>::
SubtractGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
             const std::shared_ptr<Gate<ShrType>>& p_input_y)
    : Gate<ShrType>(p_input_x, p_input_y) {
    if (p_input_x->dim_row() != p_input_y->dim_row() ||
        p_input_x->dim_col() != p_input_y->dim_col()) {
        throw std::invalid_argument("The inputs of subtraction gate should have the same dimensions");
    }
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}


template <IsSpdz2kShare ShrType>
void SubtractGate<ShrType>::doReadOfflineFromFile() {} // do nothing


template <IsSpdz2kShare ShrType>
void SubtractGate<ShrType>::doRunOnline() {
    this->Delta_clear() = matrixSubtract(this->input_x()->Delta_clear(), this->input_y()->Delta_clear());
}

} // md_ml

#endif //SUBTRACTGATE_H

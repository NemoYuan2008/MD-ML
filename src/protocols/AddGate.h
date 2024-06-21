// By Boshi Yuan

#ifndef MD_ML_ADDGATE_H
#define MD_ML_ADDGATE_H

#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class AddGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    AddGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
            const std::shared_ptr<Gate<ShrType>>& p_input_y);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;
};


template <IsSpdz2kShare ShrType>
AddGate<ShrType>::AddGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                          const std::shared_ptr<Gate<ShrType>>& p_input_y)
    : Gate<ShrType>(p_input_x, p_input_y) {
    if (p_input_x->dim_row() != p_input_y->dim_row() ||
        p_input_x->dim_col() != p_input_y->dim_col()) {
        throw std::invalid_argument("The inputs of addition gate should have the same dimensions");
    }
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}


template <IsSpdz2kShare ShrType>
void AddGate<ShrType>::doReadOfflineFromFile() {
    auto size = this->dim_row() * this->dim_col();
    this->lambda_shr() = this->party().ReadShares(size);
    this->lambda_shr_mac() = this->party().ReadShares(size);
}


template <IsSpdz2kShare ShrType>
void AddGate<ShrType>::doRunOnline() {
    this->Delta_clear() = matrixAdd(this->input_x()->Delta_clear(), this->input_y()->Delta_clear());
}

} // namespace md_ml

#endif //MD_ML_ADDGATE_H

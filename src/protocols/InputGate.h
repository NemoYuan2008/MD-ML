// By Boshi Yuan

#ifndef INPUTGATE_H
#define INPUTGATE_H

#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class InputGate : public Gate<ShrType> {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    InputGate(PartyWithFakeOffline<ShrType>& p_party,
              std::size_t p_dim_row, std::size_t p_dim_col,
              std::size_t p_owner_id);

    void setInput(const std::vector<ClearType>& input_value);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    std::size_t owner_id_;
    std::vector<SemiShrType> lambda_clear_;
    std::vector<SemiShrType> input_value_;
};


template <IsSpdz2kShare ShrType>
InputGate<ShrType>::InputGate(PartyWithFakeOffline<ShrType>& p_party,
                              std::size_t p_dim_row, std::size_t p_dim_col,
                              std::size_t p_owner_id)
    : Gate<ShrType>(p_party, p_dim_row, p_dim_col), owner_id_(p_owner_id) {}


template <IsSpdz2kShare ShrType>
void InputGate<ShrType>::
setInput(const std::vector<ClearType>& input_value) {
    if (this->party().my_id() != owner_id_)
        throw std::logic_error("Not the owner of input gate, cannot set input");
    if (input_value.size() != this->dim_row() * this->dim_col())
        throw std::invalid_argument("Input vector and gate doesn't match in size");

    input_value_ = std::vector<SemiShrType>(input_value.begin(), input_value.end());
}


template <IsSpdz2kShare ShrType>
void InputGate<ShrType>::doReadOfflineFromFile() {
    auto size = this->dim_row() * this->dim_col();

    if (this->party().my_id() == owner_id_) {
        this->lambda_clear_ = this->party().ReadShares(size);
    }

    this->lambda_shr() = this->party().ReadShares(size);
    this->lambda_shr_mac() = this->party().ReadShares(size);
}


template <IsSpdz2kShare ShrType>
void InputGate<ShrType>::doRunOnline() {
    if (this->my_id() == owner_id_) {
        this->Delta_clear() = matrixAdd(input_value_, this->lambda_clear_);
        this->party().SendVecToOther(this->Delta_clear());
    }
    else {
        auto size = this->dim_row() * this->dim_col();
        this->Delta_clear() = this->party().template ReceiveVecFromOther<SemiShrType>(size);
    }
}

} // namespace md_ml

#endif //INPUTGATE_H

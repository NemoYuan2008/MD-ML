// By Boshi Yuan

#ifndef OUTPUTGATE_H
#define OUTPUTGATE_H

#include <memory>
#include <vector>
#include <thread>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"


namespace md_ml {

template <IsSpdz2kShare ShrType>
class OutputGate : public Gate<ShrType> {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    explicit OutputGate(const std::shared_ptr<Gate<ShrType>>& p_input_x);

    std::vector<ClearType> getClear() const;

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    std::vector<SemiShrType> lambda_clear_;
    std::vector<SemiShrType> output_value_;
};


template <IsSpdz2kShare ShrType>
OutputGate<ShrType>::OutputGate(const std::shared_ptr<Gate<ShrType>>& p_input_x)
    : Gate<ShrType>(p_input_x, nullptr) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}


template <IsSpdz2kShare ShrType>
void OutputGate<ShrType>::doReadOfflineFromFile() {} // Do nothing


template <IsSpdz2kShare ShrType>
void OutputGate<ShrType>::doRunOnline() {
    auto size = this->dim_row() * this->dim_col();

    std::thread t1([this] {
        this->party().SendVecToOther(this->input_x()->lambda_shr());
    });

    std::thread t2([this, size] {
        this->lambda_clear_ = this->party().template ReceiveVecFromOther<SemiShrType>(size);
    });

    t1.join();
    t2.join();

    matrixAddAssign(lambda_clear_, this->input_x()->lambda_shr()); // reconstruct $\lambda_x$
    output_value_ = matrixSubtract(this->input_x()->Delta_clear(), lambda_clear_); // $x = \Delta_x - \lambda_x$
}


template <IsSpdz2kShare ShrType>
std::vector<typename OutputGate<ShrType>::ClearType> OutputGate<ShrType>::
getClear() const {
    return std::vector<ClearType>(output_value_.begin(), output_value_.end());
}

} // md_ml

#endif //OUTPUTGATE_H

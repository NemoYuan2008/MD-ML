// By Boshi Yuan

#ifndef MD_ML_MULTIPLYGATE_H
#define MD_ML_MULTIPLYGATE_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <thread>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class MultiplyGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    MultiplyGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                 const std::shared_ptr<Gate<ShrType>>& p_input_y);

    [[nodiscard]] std::size_t dim_mid() const { return dim_mid_; }

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    std::size_t dim_mid_;

    std::vector<SemiShrType> a_shr_, a_shr_mac_;
    std::vector<SemiShrType> b_shr_, b_shr_mac_;
    std::vector<SemiShrType> c_shr_, c_shr_mac_;
    std::vector<SemiShrType> delta_x_clear_;
    std::vector<SemiShrType> delta_y_clear_;
};

template <IsSpdz2kShare ShrType>
MultiplyGate<ShrType>::
MultiplyGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
             const std::shared_ptr<Gate<ShrType>>& p_input_y)
    : Gate<ShrType>(p_input_x, p_input_y), dim_mid_(p_input_x->dim_col()) {
    // check and set dimensions
    if (p_input_x->dim_col() != p_input_y->dim_row()) {
        throw std::invalid_argument("The inputs of multiplication gate should have compatible dimensions");
    }
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_y->dim_col());
    // dim_mid_ was set in the initializer list
}

template <IsSpdz2kShare ShrType>
void MultiplyGate<ShrType>::doReadOfflineFromFile() {
    auto size_lhs = this->dim_row() * this->dim_mid();
    auto size_rhs = this->dim_mid() * this->dim_col();
    auto size_output = this->dim_row() * this->dim_col();

    a_shr_ = this->party().ReadShares(size_lhs);
    a_shr_mac_ = this->party().ReadShares(size_lhs);
    b_shr_ = this->party().ReadShares(size_rhs);
    b_shr_mac_ = this->party().ReadShares(size_rhs);
    c_shr_ = this->party().ReadShares(size_output);
    c_shr_mac_ = this->party().ReadShares(size_output);
    this->lambda_shr() = this->party().ReadShares(size_output);
    this->lambda_shr_mac() = this->party().ReadShares(size_output);
    delta_x_clear_ = this->party().ReadShares(size_lhs);
    delta_y_clear_ = this->party().ReadShares(size_rhs);
}

template <IsSpdz2kShare ShrType>
void MultiplyGate<ShrType>::doRunOnline() {
    // temp_x = $\Delta_x + \delta_x$
    auto temp_x = matrixAdd(this->input_x()->Delta_clear(), delta_x_clear_);
    // temp_y = $\Delta_y + \delta_y$
    auto temp_y = matrixAdd(this->input_y()->Delta_clear(), delta_y_clear_);
    // temp_xy = temp_x * temp_y
    auto temp_xy = matrixMultiply(temp_x, temp_y, this->dim_row(), this->dim_mid(), this->dim_col());


    // Compute [Delta_z] according to the paper
    // [Delta_z] = [c] + [lambda_z]
    auto Delta_z_shr = matrixAdd(c_shr_, this->lambda_shr());
    // [Delta_z] -= [a] * temp_y
    matrixSubtractAssign(Delta_z_shr,
                         matrixMultiply(a_shr_, temp_y, this->dim_row(), this->dim_mid(), this->dim_col()));
    // [Delta_z] -= temp_x * [b]
    matrixSubtractAssign(Delta_z_shr,
                         matrixMultiply(temp_x, b_shr_, this->dim_row(), this->dim_mid(), this->dim_col()));
    if (this->my_id() == 0) {
        // [Delta_z] += temp_xy
        matrixAddAssign(Delta_z_shr, temp_xy);
    }

    // Compute Delta_z_mac according to the paper
    // [Delta_z_mac] = temp_xy * [key]
    auto Delta_z_mac = std::move(temp_xy);
    matrixScalarAssign(Delta_z_mac, this->party().global_key_shr());
    // [Delta_z_mac] += [c_mac] + [lambda_z_mac]
    matrixAddAssign(Delta_z_mac,
                    matrixAdd(c_shr_mac_, this->lambda_shr_mac()));
    // [Delta_z_mac] -= [a_mac] * temp_y_mac
    matrixSubtractAssign(Delta_z_mac,
                         matrixMultiply(a_shr_mac_, temp_y, this->dim_row(), this->dim_mid(), this->dim_col()));
    // [Delta_z_mac] -= temp_x_mac * [b_mac]
    matrixSubtractAssign(Delta_z_mac,
                         matrixMultiply(temp_x, b_shr_mac_, this->dim_row(), this->dim_mid(), this->dim_col()));

    std::thread t1([&, this] {
        this->party().SendVecToOther(Delta_z_shr);
    });

    std::thread t2([&, this] {
        this->Delta_clear() = this->party().template ReceiveVecFromOther<SemiShrType>(this->dim_row() * this->dim_col());
    });

    t1.join();
    t2.join();

    matrixAddAssign(this->Delta_clear(), Delta_z_shr);

    // free the spaces of preprocessing data
    a_shr_.clear();
    a_shr_.shrink_to_fit();
    a_shr_mac_.clear();
    a_shr_mac_.shrink_to_fit();
    b_shr_.clear();
    b_shr_.shrink_to_fit();
    b_shr_mac_.clear();
    b_shr_mac_.shrink_to_fit();
    c_shr_.clear();
    c_shr_.shrink_to_fit();
    c_shr_mac_.clear();
    c_shr_mac_.shrink_to_fit();
    delta_x_clear_.clear();
    delta_x_clear_.shrink_to_fit();
    delta_y_clear_.clear();
    delta_y_clear_.shrink_to_fit();
}

} // md_ml

#endif //MD_ML_MULTIPLYGATE_H

// By Boshi Yuan

#ifndef MD_ML_MULTIPLYTRUNCGATE_H
#define MD_ML_MULTIPLYTRUNCGATE_H


#include <memory>
#include <vector>
#include <stdexcept>
#include <thread>

#include "protocols/Gate.h"
#include "protocols/MultiplyGate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"
#include "utils/fixed_point.h"


namespace md_ml {

template <IsSpdz2kShare ShrType>
class MultiplyTruncGate : public MultiplyGate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    MultiplyTruncGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                      const std::shared_ptr<Gate<ShrType>>& p_input_y);

protected:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

private:
    std::vector<SemiShrType> lambda_prime_shr_;
    std::vector<SemiShrType> lambda_prime_shr_mac_;
};


template <IsSpdz2kShare ShrType>
MultiplyTruncGate<ShrType>::
MultiplyTruncGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                  const std::shared_ptr<Gate<ShrType>>& p_input_y)
    : MultiplyGate<ShrType>(p_input_x, p_input_y) {
    // Check and set dimensions
    if (p_input_x->dim_col() != p_input_y->dim_row()) {
        throw std::invalid_argument("The inputs of multiplication gate should have compatible dimensions");
    }
}


template <IsSpdz2kShare ShrType>
void MultiplyTruncGate<ShrType>::doReadOfflineFromFile() {
    MultiplyGate<ShrType>::doReadOfflineFromFile();
    auto size_output = this->dim_row() * this->dim_col();

    // The lambda_shr_ in the base class is actually lambda_prime_shr_ here, so is the lambda_shr_mac_.
    // So we compute the real lambda_shr_ in lambda_prime_shr_, and the real lambda_shr_mac_ in lambda_prime_shr_mac_.
    // Then swap the corresponding vectors.
    lambda_prime_shr_ = this->party().ReadShares(size_output);
    lambda_prime_shr_mac_ = this->party().ReadShares(size_output);
}


template <IsSpdz2kShare ShrType>
void MultiplyTruncGate<ShrType>::doRunOnline() {
    MultiplyGate<ShrType>::doRunOnline();

    // The swap is done after the computation of the multiplication,
    // because the real prime values are used in the protocol.
    lambda_prime_shr_.swap(this->lambda_shr());
    lambda_prime_shr_mac_.swap(this->lambda_shr_mac());

    // Delta_z = Delta_z' / 2^d
    truncateClearVecInplace(this->Delta_clear());
}

} // namespace md_ml

#endif //MD_ML_MULTIPLYTRUNCGATE_H

// By Boshi Yuan

#ifndef CONV2DTRUNCGATE_H
#define CONV2DTRUNCGATE_H


#include <memory>
#include <vector>

#include "protocols/Gate.h"
#include "protocols/Conv2DGate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/fixed_point.h"


namespace md_ml {

template <IsSpdz2kShare ShrType>
class Conv2DTruncGate : public Conv2DGate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    Conv2DTruncGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                    const std::shared_ptr<Gate<ShrType>>& p_input_y,
                    const Conv2DOp& op);

protected:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

private:
    std::vector<SemiShrType> lambda_prime_shr_;
    std::vector<SemiShrType> lambda_prime_shr_mac_;
};

template <IsSpdz2kShare ShrType>
Conv2DTruncGate<ShrType>::
Conv2DTruncGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                const std::shared_ptr<Gate<ShrType>>& p_input_y,
                const Conv2DOp& op)
    : Conv2DGate<ShrType>(p_input_x, p_input_y, op) {
    // The dimensions are already set in the base class
}

template <IsSpdz2kShare ShrType>
void Conv2DTruncGate<ShrType>::doReadOfflineFromFile() {
    Conv2DGate<ShrType>::doReadOfflineFromFile();

    auto size_output = this->conv_op().compute_output_size();

    // The lambda_shr_ in the base class is actually lambda_prime_shr_ here, so is the lambda_shr_mac_.
    // So we compute the real lambda_shr_ in lambda_prime_shr_, and the real lambda_shr_mac_ in lambda_prime_shr_mac_.
    // Then swap the corresponding vectors.
    lambda_prime_shr_ = this->party().ReadShares(size_output);
    lambda_prime_shr_mac_ = this->party().ReadShares(size_output);
}

template <IsSpdz2kShare ShrType>
void Conv2DTruncGate<ShrType>::doRunOnline() {
    Conv2DGate<ShrType>::doRunOnline();

    // The swap is done after the computation of the multiplication,
    // because the real prime values are used in the protocol.
    lambda_prime_shr_.swap(this->lambda_shr());
    lambda_prime_shr_mac_.swap(this->lambda_shr_mac());

    // Delta_z = Delta_z' / 2^d
    truncateClearVecInplace(this->Delta_clear());
}

} // namespace md_ml

#endif //CONV2DTRUNCGATE_H

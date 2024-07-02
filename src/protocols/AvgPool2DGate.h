// By Boshi Yuan

#ifndef AVGPOOL2DGATE_H
#define AVGPOOL2DGATE_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <thread>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"
#include "utils/tensor.h"
#include "utils/fixed_point.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class AvgPool2DGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;
    using ClearType = typename ShrType::ClearType;

    AvgPool2DGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
                  const MaxPoolOp& op);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    MaxPoolOp maxPoolOp;
    ClearType factor; // equals 1 / kernel_size
    std::vector<SemiShrType> lambdaPreTruncShr, lambdaPreTruncShrMac;
};

template <IsSpdz2kShare ShrType>
AvgPool2DGate<ShrType>::
AvgPool2DGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
              const MaxPoolOp& op)
    : Gate<ShrType>(p_input_x, nullptr),
      maxPoolOp(op),
      factor(double2fix<ClearType>(1.0 / op.compute_kernel_size())) {
    this->set_dim_row(maxPoolOp.compute_output_size());
    this->set_dim_col(1);
}

template <IsSpdz2kShare ShrType>
void AvgPool2DGate<ShrType>::doReadOfflineFromFile() {
    auto size = this->maxPoolOp.compute_output_size();
    this->lambda_shr() = this->party().ReadShares(size);
    this->lambda_shr_mac() = this->party().ReadShares(size);
    lambdaPreTruncShr = this->party().ReadShares(size);
    lambdaPreTruncShrMac = this->party().ReadShares(size);
}

template <IsSpdz2kShare ShrType>
void AvgPool2DGate<ShrType>::doRunOnline() {
    const auto& delta_x_clear = this->input_x()->Delta_clear();
    const auto& lambda_x_shr = this->input_x()->lambda_shr();

    // [x] = Delta_x - [lambda_x]
    std::vector<SemiShrType> x_shr;
    if (this->my_id() == 0) {
        x_shr = matrixSubtract(delta_x_clear, lambda_x_shr);
    }
    else {
        x_shr.resize(lambda_x_shr.size());
        std::transform(lambda_x_shr.begin(), lambda_x_shr.end(), x_shr.begin(), std::negate());
    }

    auto size = maxPoolOp.compute_output_size();

    auto delta_zShr = sumPool(x_shr, maxPoolOp);
    matrixScalarAssign(delta_zShr, static_cast<SemiShrType>(factor));

    assert(delta_zShr.size() == size);

    //truncation (needs communication)
    matrixAddAssign(delta_zShr, lambdaPreTruncShr);

    std::vector<SemiShrType> delta_zRcv(size);
    std::thread t1([this, &delta_zShr] {
        this->party().SendVecToOther(delta_zShr);
    });
    std::thread t2([this, &delta_zRcv, size] {
        delta_zRcv = this->party().template ReceiveVecFromOther<SemiShrType>(size);
    });
    t1.join();
    t2.join();

    this->Delta_clear() = matrixAdd(delta_zShr, delta_zRcv);
    truncateClearVecInplace(this->Delta_clear());
}

} // namespace md_ml

#endif //AVGPOOL2DGATE_H

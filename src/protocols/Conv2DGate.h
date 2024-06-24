// By Boshi Yuan

#ifndef MD_ML_CONV2DGATE_H
#define MD_ML_CONV2DGATE_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <thread>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"
#include "utils/tensor.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class Conv2DGate : public Gate<ShrType> {
public:
    using SemiShrType = typename ShrType::SemiShrType;

    Conv2DGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
               const std::shared_ptr<Gate<ShrType>>& p_input_y,
               const Conv2DOp& op);

};

template <IsSpdz2kShare ShrType>
Conv2DGate<ShrType>::
Conv2DGate(const std::shared_ptr<Gate<ShrType>>& p_input_x,
           const std::shared_ptr<Gate<ShrType>>& p_input_y,
           const Conv2DOp& op)
    : Gate<ShrType>(p_input_x, p_input_y) {
    this->set_dim_row(op.compute_output_size());
    this->set_dim_col(1);
}

}


#endif //MD_ML_CONV2DGATE_H

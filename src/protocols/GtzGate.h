// By Shixuan Yang

#ifndef GTZGATE_H
#define GTZGATE_H

#include <memory>
#include <vector>
#include <bitset>
#include <thread>

#include "protocols/Gate.h"
#include "share/IsSpdz2kShare.h"
#include "utils/linear_algebra.h"


namespace md_ml {

template <IsSpdz2kShare ShrType>
class GtzGate : public Gate<ShrType> {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    explicit GtzGate(const std::shared_ptr<Gate<ShrType>>& p_input_x);

private:
    void doReadOfflineFromFile() override;
    void doRunOnline() override;

    std::vector<bool> BitLT(std::vector<ClearType>& pInt,
                            std::vector<ClearType>& sInt);

    std::vector<bool> CarryOutCin(std::vector<std::bitset<sizeof(ClearType) * 8>>& aIn,
                                  std::vector<std::bitset<sizeof(ClearType) * 8>>& bIn,
                                  bool cIn);

    template <typename T>
    bool CarryOutAux(std::bitset<sizeof(T) * 8> p,
                     std::bitset<sizeof(T) * 8> g, int k);

    std::vector<ClearType> lambda_xBinShr;
    bool a = false, b = false, c = false; // Binary Triples, we currently fake them as (0, 0, 0)
};


template <IsSpdz2kShare ShrType>
GtzGate<ShrType>::
GtzGate(const std::shared_ptr<Gate<ShrType>>& p_input_x)
    : Gate<ShrType>(p_input_x, nullptr) {
    this->set_dim_row(p_input_x->dim_row());
    this->set_dim_col(p_input_x->dim_col());
}

template <IsSpdz2kShare ShrType>
void GtzGate<ShrType>::doReadOfflineFromFile() {
    auto size = this->dim_row() * this->dim_col();

    this->lambda_shr() = this->party().ReadShares(size);
    this->lambda_shr_mac() = this->party().ReadShares(size);
    lambda_xBinShr = this->party().ReadClear(size);
}

template <IsSpdz2kShare ShrType>
void GtzGate<ShrType>::doRunOnline() {}

} // namespace md_ml

#endif //GTZGATE_H

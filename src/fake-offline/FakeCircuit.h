// By Boshi Yuan

#ifndef MD_ML_FAKECIRCUIT_H
#define MD_ML_FAKECIRCUIT_H


#include <memory>
#include <vector>
#include <cstddef>

#include "share/IsSpdz2kShare.h"
#include "fake-offline/FakeParty.h"
#include "fake-offline/FakeGate.h"
#include "fake-offline/FakeInputGate.h"
#include "fake-offline/FakeAddGate.h"
#include "fake-offline/FakeSubtractGate.h"
#include "fake-offline/FakeMultiplyGate.h"


namespace md_ml {


/// @brief A fake circuit that uses the fake offline protocol to compute the result
/// @tparam ShrType
/// @tparam N
template <IsSpdz2kShare ShrType, std::size_t N>
class FakeCircuit {
public:
    explicit FakeCircuit(FakeParty<ShrType, N>& p_fake_party) : fake_party_(p_fake_party) {}

    void runOffline() {
        for (const auto& gatePtr : endpoints_) {
            gatePtr->runOffline();
        }
    }

    void addEndpoint(const std::shared_ptr<FakeGate<ShrType, N>> &gate) {
        endpoints_.push_back(gate);
    }

    std::shared_ptr<FakeInputGate<ShrType, N>>
    input(int owner_id = 0, std::size_t dim_row = 1, std::size_t dim_col = 1) {
        auto gate = std::make_shared<FakeInputGate<ShrType, N>>(fake_party_, dim_row, dim_col, owner_id);
        gates_.push_back(gate);
        return gate;
    }

    std::shared_ptr<FakeAddGate<ShrType, N>>
    add(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
        const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
        auto gate = std::make_shared<FakeAddGate<ShrType, N>>(input_x, input_y);
        gates_.push_back(gate);
        return gate;
    }

    std::shared_ptr<FakeSubtractGate<ShrType, N>>
    subtract(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
             const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
        auto gate = std::make_shared<FakeSubtractGate<ShrType, N>>(input_x, input_y);
        gates_.push_back(gate);
        return gate;
    }

    std::shared_ptr<FakeMultiplyGate<ShrType, N>>
    multiply(const std::shared_ptr<FakeGate<ShrType, N>>& input_x,
             const std::shared_ptr<FakeGate<ShrType, N>>& input_y) {
        auto gate = std::make_shared<FakeMultiplyGate<ShrType, N>>(input_x, input_y);
        gates_.push_back(gate);
        return gate;
    }

private:
    FakeParty<ShrType, N>& fake_party_;
    std::vector<std::shared_ptr<FakeGate<ShrType, N>>> gates_;
    std::vector<std::shared_ptr<FakeGate<ShrType, N>>> endpoints_;
};


} // namespace md_ml

#endif //MD_ML_FAKECIRCUIT_H

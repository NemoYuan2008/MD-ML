// By Boshi Yuan

#ifndef FAKEGATE_H
#define FAKEGATE_H

#include <memory>
#include <vector>
#include <array>
#include <cstddef>

#include "share/IsSpdz2kShare.h"
#include "utils/uint128_io.h"
#include "fake-offline/FakeParty.h"


namespace md_ml {

template <IsSpdz2kShare ShrType, std::size_t N>
class FakeGate {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    FakeGate(FakeParty<ShrType, N>& p_fake_party, std::size_t p_dim_row, std::size_t p_dim_col)
        : fake_party_(p_fake_party), dim_row_(p_dim_row), dim_col_(p_dim_col) {}

    FakeGate(const std::shared_ptr<FakeGate>& p_input_x,
             const std::shared_ptr<FakeGate>& p_input_y)
        : fake_party_(p_input_x->fake_party_), input_x_(p_input_x), input_y_(p_input_y) {}

    virtual ~FakeGate() = default;

    void runOffline();

    [[nodiscard]] bool isEvaluatedOffline() const { return evaluated_offline_; }

    [[nodiscard]] const auto& input_x() const { return input_x_; }

    [[nodiscard]] const auto& input_y() const { return input_y_; }

    [[nodiscard]] auto dim_row() const { return dim_row_; }

    [[nodiscard]] auto dim_col() const { return dim_col_; }

    [[nodiscard]] auto& fake_party() { return fake_party_; }
    [[nodiscard]] const auto& fake_party() const { return fake_party_; }

    [[nodiscard]] auto& lambda_clear() { return lambda_clear_; }
    [[nodiscard]] const auto& lambda_clear() const { return lambda_clear_; }

    [[nodiscard]] auto& lambda_shr() { return lambda_shr_; }
    [[nodiscard]] const auto& lambda_shr() const { return lambda_shr_; }

    [[nodiscard]] auto& lambda_shr_mac() { return lambda_shr_mac_; }
    [[nodiscard]] const auto& lambda_shr_mac() const { return lambda_shr_mac_; }

protected:
    void set_dim_row(std::size_t p_dim_row) { dim_row_ = p_dim_row; }
    void set_dim_col(std::size_t p_dim_col) { dim_col_ = p_dim_col; }

private:
    virtual void doRunOffline() = 0;

    // Since circuit evaluation is done in a recursive manner,
    // we need to keep track of whether the gate has been evaluated
    bool evaluated_offline_ = false;

    FakeParty<ShrType, N>& fake_party_;

    // The inputs wires of the gate
    std::shared_ptr<FakeGate> input_x_{};
    std::shared_ptr<FakeGate> input_y_{};

    // A gate actually holds a matrix, not a single value
    // The values are stored in a flat std::vector, so we store the dimensions of the matrix
    std::size_t dim_row_ = 1;
    std::size_t dim_col_ = 1;

    // The $\lambda_z$ values in the clear
    std::vector<ClearType> lambda_clear_;

    // The shares of $\lambda_z$-values held by the parties
    std::array<std::vector<SemiShrType>, N> lambda_shr_;

    // The shared MAC of the $\lambda_z$-values held by the parties
    std::array<std::vector<SemiShrType>, N> lambda_shr_mac_;
};


template <IsSpdz2kShare ShrType, std::size_t N>
void FakeGate<ShrType, N>::runOffline() {
    if (this->isEvaluatedOffline())
        return;

    if (input_x_ && !input_x_->isEvaluatedOffline())
        input_x_->runOffline();
    if (input_y_ && !input_y_->isEvaluatedOffline())
        input_y_->runOffline();

    this->doRunOffline();

    this->evaluated_offline_ = true;
}

} // namespace md_ml

#endif //FAKEGATE_H

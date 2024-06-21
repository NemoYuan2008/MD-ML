// By Boshi Yuan

#ifndef MD_ML_PROTOCOLS_GATE_H
#define MD_ML_PROTOCOLS_GATE_H


#include <vector>
#include <memory>

#include "networking/Party.h"
#include "share/IsSpdz2kShare.h"
#include "protocols/PartyWithFakeOffline.h"


namespace md_ml {


template <IsSpdz2kShare ShrType>
class Gate {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    Gate(PartyWithFakeOffline<ShrType>& p_party, std::size_t p_dim_row, std::size_t p_dim_col);

    Gate(const std::shared_ptr<Gate>& p_input_x, const std::shared_ptr<Gate>& p_input_y);

    virtual ~Gate() = default;

    void RunOffline();
    void readOfflineFromFile();
    void RunOnline();

    [[nodiscard]] auto& party() { return party_; }

    [[nodiscard]] std::size_t my_id() const { return party_.my_id(); }

    [[nodiscard]] std::size_t dim_row() const { return dim_row_; }
    [[nodiscard]] std::size_t dim_col() const { return dim_col_; }

    [[nodiscard]] auto input_x() { return input_x_; }
    [[nodiscard]] auto input_y() { return input_y_; }

    [[nodiscard]] const std::vector<SemiShrType>& lambda_shr() const { return lambda_shr_; }
    [[nodiscard]] std::vector<SemiShrType>& lambda_shr() { return lambda_shr_; }

    [[nodiscard]] const std::vector<SemiShrType>& lambda_shr_mac() const { return lambda_shr_mac_; }
    [[nodiscard]] std::vector<SemiShrType>& lambda_shr_mac() { return lambda_shr_mac_; }

    [[nodiscard]] const std::vector<SemiShrType>& Delta_clear() const { return Delta_clear_; }
    [[nodiscard]] std::vector<SemiShrType>& Delta_clear() { return Delta_clear_; }

protected:
    void set_dim_row(std::size_t p_dim_row) { dim_row_ = p_dim_row; }
    void set_dim_col(std::size_t p_dim_col) { dim_col_ = p_dim_col; }

private:
    virtual void doRunOffline() { throw std::runtime_error("Offline Phase is not implemented."); }
    virtual void doReadOfflineFromFile() = 0;
    virtual void doRunOnline() = 0;

    bool evaluated_offline_ = false;
    bool evaluated_online_ = false;
    bool read_offline_ = false;

    PartyWithFakeOffline<ShrType>& party_;

    // The input wires of the gate
    std::shared_ptr<Gate> input_x_{};
    std::shared_ptr<Gate> input_y_{};

    // A gate actually holds a matrix, not a single value
    // The values are stored in a flat std::vector, so we store the dimensions of the matrix
    std::size_t dim_row_ = 1;
    std::size_t dim_col_ = 1;

    std::vector<SemiShrType> lambda_shr_;
    std::vector<SemiShrType> lambda_shr_mac_;
    std::vector<SemiShrType> Delta_clear_; // We store it as SemiShrType for convienence, but it's actually ClearType
};


template <IsSpdz2kShare ShrType>
Gate<ShrType>::Gate(PartyWithFakeOffline<ShrType>& p_party, std::size_t p_dim_row, std::size_t p_dim_col)
    : party_(p_party), dim_row_(p_dim_row), dim_col_(p_dim_col) {}


template <IsSpdz2kShare ShrType>
Gate<ShrType>::Gate(const std::shared_ptr<Gate>& p_input_x, const std::shared_ptr<Gate>& p_input_y)
    : party_(p_input_x->party()), input_x_(p_input_x), input_y_(p_input_y) {}


template <IsSpdz2kShare ShrType>
void Gate<ShrType>::RunOffline() {
    if (this->evaluated_offline_)
        return;

    if (input_x_ && !input_x_->evaluated_offline_)
        input_x_->runOffline();
    if (input_y_ && !input_y_->evaluated_offline_)
        input_y_->runOffline();

    this->doRunOffline();

    this->evaluated_offline_ = true;
}


template <IsSpdz2kShare ShrType>
void Gate<ShrType>::readOfflineFromFile() {
    if (this->read_offline_)
        return;

    if (input_x_ && !input_x_->read_offline_)
        input_x_->readOfflineFromFile();
    if (input_y_ && !input_y_->read_offline_)
        input_y_->readOfflineFromFile();

    this->doReadOfflineFromFile();

    this->read_offline_ = true;
}


template <IsSpdz2kShare ShrType>
void Gate<ShrType>::RunOnline() {
    if (this->evaluated_online_)
        return;

    if (input_x_ && !input_x_->evaluated_online_)
        input_x_->RunOnline();
    if (input_y_ && !input_y_->evaluated_online_)
        input_y_->RunOnline();

    this->doRunOnline();

    this->evaluated_online_ = true;
}

}


#endif //MD_ML_PROTOCOLS_GATE_H

// By Boshi Yuan

#ifndef MD_ML_PROTOCOLS_GATE_H
#define MD_ML_PROTOCOLS_GATE_H


#include <vector>
#include <memory>

#include "networking/Party.h"
#include "share/IsSpdz2kShare.h"


namespace md_ml {


template <IsSpdz2kShare ShrType>
class Gate {
public:
    using ClearType = typename ShrType::ClearType;
    using SemiShrType = typename ShrType::SemiShrType;

    Gate(Party& p_party, std::size_t p_dim_row, std::size_t p_dim_col);

    Gate(const std::shared_ptr<Gate>& p_input_x,
         const std::shared_ptr<Gate>& p_input_y);

    virtual ~Gate() = default;

    void RunOffline();

    void RunOnline();

    void readOfflineFromFile();

    [[nodiscard]] Party& party() { return party_; }

    [[nodiscard]] std::vector<SemiShrType>& lambda_shr() { return lambda_shr_; }

    [[nodiscard]] std::vector<SemiShrType>& lambda_shr_mac() { return lambda_shr_mac_; }

    [[nodiscard]] std::vector<ClearType>& delta_clear() { return Delta_clear; }

protected:
    void set_dim_row(std::size_t p_dim_row) { dim_row_ = p_dim_row; }
    void set_dim_col(std::size_t p_dim_col) { dim_col_ = p_dim_col; }

private:
    virtual void doRunOnline() = 0;
    virtual void doRunOffline() { throw std::runtime_error("Offline Phase is not implemented."); }
    virtual void doReadOfflineFromFile() = 0;

    bool evaluated_offline_ = false;
    bool evaluated_online_ = false;
    bool read_offline_ = false;

    Party& party_;

    // The input wires of the gate
    std::shared_ptr<Gate> input_x_{};
    std::shared_ptr<Gate> input_y_{};

    // A gate actually holds a matrix, not a single value
    // The values are stored in a flat std::vector, so we store the dimensions of the matrix
    std::size_t dim_row_ = 1;
    std::size_t dim_col_ = 1;

    std::vector<SemiShrType> lambda_shr_;
    std::vector<SemiShrType> lambda_shr_mac_;
    std::vector<ClearType> Delta_clear;
};


template <IsSpdz2kShare ShrType>
Gate<ShrType>::Gate(Party& p_party, std::size_t p_dim_row, std::size_t p_dim_col)
    : party_(p_party), dim_row_(p_dim_row), dim_col_(p_dim_col) {}


template <IsSpdz2kShare ShrType>
Gate<ShrType>::Gate(const std::shared_ptr<Gate>& p_input_x, const std::shared_ptr<Gate>& p_input_y)
    : party_(p_input_x->party()), input_x_(p_input_x), input_y_(p_input_y) {}


// template <IsSpdz2kShare ShrType>
// void Gate<ShrType>::RunOffline() {}
//
// template <IsSpdz2kShare ShrType>
// void Gate<ShrType>::RunOnline() {}
//
// template <IsSpdz2kShare ShrType>
// void Gate<ShrType>::readOfflineFromFile() {}

}


#endif //MD_ML_PROTOCOLS_GATE_H

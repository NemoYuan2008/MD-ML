// By Boshi Yuan

#ifndef MD_ML_PartyWithFakeOffline_H
#define MD_ML_PartyWithFakeOffline_H


#include <string>
#include <fstream>
#include <filesystem>

#include "share/IsSpdz2kShare.h"
#include "networking/Party.h"
#include "utils/uint128_io.h"

namespace md_ml {

template <IsSpdz2kShare ShrType>
class PartyWithFakeOffline : public Party {
public:
    using ClearType = typename ShrType::ClearType;
    using KeyShrType = typename ShrType::KeyShrType;
    using GlobalKeyType = typename ShrType::GlobalKeyType;
    using SemiShrType = typename ShrType::SemiShrType;

    PartyWithFakeOffline(std::size_t p_my_id, std::size_t p_num_parties, std::size_t p_port,
                         const std::string& job_name);

    std::vector<SemiShrType> ReadShares(std::size_t num_elements);

    std::vector<ClearType> ReadClear(std::size_t num_elements);

    [[nodiscard]] std::ifstream& input_file() { return input_file_; }

    [[nodiscard]] GlobalKeyType global_key_shr() const { return global_key_shr_; }

private:
    inline static const std::filesystem::path kFakeOfflineDir{FAKE_OFFLINE_DIR}; // The macro is in CMakeLists.txt
    GlobalKeyType global_key_shr_;
    std::ifstream input_file_;
};


template <IsSpdz2kShare ShrType>
PartyWithFakeOffline<ShrType>::
PartyWithFakeOffline(std::size_t p_my_id, std::size_t p_num_parties, std::size_t p_port, const std::string& job_name)
    : Party(p_my_id, p_num_parties, p_port) {
    // Open the file for input
    std::string file_name = job_name + (job_name.empty() ? "party-" : "-party-") + std::to_string(p_my_id) + ".txt";
    input_file_.open(kFakeOfflineDir / file_name);

    // Read the MAC key
    input_file_ >> global_key_shr_;
}


template <IsSpdz2kShare ShrType>
std::vector<typename PartyWithFakeOffline<ShrType>::SemiShrType> PartyWithFakeOffline<ShrType>::
ReadShares(std::size_t num_elements) {
    auto shares = std::vector<SemiShrType>(num_elements);
    for (auto& share : shares) {
        input_file_ >> share;
    }
    return shares;
}

template <IsSpdz2kShare ShrType>
std::vector<typename PartyWithFakeOffline<ShrType>::ClearType> PartyWithFakeOffline<ShrType>::
ReadClear(std::size_t num_elements) {
    auto clear = std::vector<ClearType>(num_elements);
    for (auto& c : clear) {
        input_file_ >> c;
    }
    return clear;
}

} // namespace md_ml


#endif //MD_ML_PartyWithFakeOffline_H

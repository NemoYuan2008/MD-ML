// By Boshi Yuan

#ifndef MD_ML_FAKEPARTY_H
#define MD_ML_FAKEPARTY_H


#include <fstream>
#include <filesystem>
#include <string>
#include <array>
#include <numeric>
#include <algorithm>
#include <cstddef>

#include "share/IsSpdz2kShare.h"
#include "utils/rand.h"

namespace md_ml {


/// A fake party that generates all preprocessing data for all parties,
/// the preprocessed data are stored in local files.
/// Note that only one FakeParty object should be created for each job.
///
/// @tparam ShrType The type of the shares, should be a Spdz2kShare<K, S> type
/// @tparam N The number of the parties
template <IsSpdz2kShare ShrType, std::size_t N>
class FakeParty {
public:
    using ClearType = typename ShrType::ClearType;
    using KeyShrType = typename ShrType::KeyShrType;
    using GlobalKeyType = typename ShrType::GlobalKeyType;
    using SemiShrType = typename ShrType::SemiShrType;

    /// A helper struct that stores the shares of a value and its MAC held by all parties. (Only for internal use.)
    struct AllPartiesShares {
        std::array<SemiShrType, N> value_shares;
        std::array<SemiShrType, N> mac_shares;
    };

    /// A vectorized version of AllPartiesShares (Only for internal use.)
    /// We don't vectorize AllPartiesShares directly, since it is consistent with the gates' output format.
    struct AllPartiesSharesVec {
        std::array<std::vector<SemiShrType>, N> value_shares;
        std::array<std::vector<SemiShrType>, N> mac_shares;
    };

    /// Constructs a FakeParty object
    /// @param job_name (optional) The name of the job, used to generate the output file names,
    ///                 if not provided, the output file names will be "party-0.txt", "party-1.txt", ...
    ///                 Otherwise, the output file names will be "<job_name>-party-0.txt", etc.
    explicit FakeParty(const std::string& job_name = std::string());

    /// Returns the number of parties
    auto constexpr static NParties() noexcept { return N; }

    /// Returns the i-th party's output ofstream
    [[nodiscard]] auto& ithPartyFile(std::size_t i) { return output_files_.at(i); }

    AllPartiesShares GenerateAllPartiesShares(ClearType value) const;

    AllPartiesSharesVec GenerateAllPartiesShares(const std::vector<ClearType>& value) const;

    void WriteSharesToAllParites(const std::array<std::vector<SemiShrType>, N>& shares,
                                 const std::array<std::vector<SemiShrType>, N>& macs);

    void WriteClearToIthParty(const std::vector<ClearType>& values, std::size_t party_id);

    void WriteClearToAllParties(const std::vector<ClearType>& values);

private:
    inline static const std::filesystem::path kFakeOfflinePath{FAKE_OFFLINE_DIR}; // The macro is in CMakeLists.txt
    GlobalKeyType global_key_;
    std::array<KeyShrType, N> key_shares_;
    std::array<std::ofstream, N> output_files_;
};


template <IsSpdz2kShare ShrType, std::size_t N>
FakeParty<ShrType, N>::FakeParty(const std::string& job_name) {
    // Open the output files for each party
    if (!exists(kFakeOfflinePath)) {
        create_directory(kFakeOfflinePath);
    }
    const std::string file_name_suffix = job_name + (job_name.empty() ? "party-" : "-party-");
    for (std::size_t i = 0; i < N; ++i) {
        std::string current_file_name = file_name_suffix + std::to_string(i) + ".txt";
        output_files_[i].open(kFakeOfflinePath / current_file_name);
    }

    // Generate the MAC key
    global_key_ = 0;
    for (std::size_t i = 0; i < N; ++i) {
        key_shares_[i] = getRand<KeyShrType>();
        global_key_ += static_cast<GlobalKeyType>(key_shares_[i]);
    }

    // Write the MAC key to the output files for each party
    for (std::size_t i = 0; i < N; ++i) {
        output_files_[i] << key_shares_[i] << '\n';
    }
}


template <IsSpdz2kShare ShrType, std::size_t N>
typename FakeParty<ShrType, N>::AllPartiesShares FakeParty<ShrType, N>::
GenerateAllPartiesShares(ClearType value) const {
    AllPartiesShares all_parties_shares;

    // The upper s bits of the value should be masked, see https://ia.cr/2018/482 Section 3
    auto mask = getRand<KeyShrType>();
    auto masked_value = static_cast<SemiShrType>(mask) << ShrType::kBits | static_cast<SemiShrType>(value);

    // mac = value * key
    SemiShrType mac = masked_value * global_key_;

    // generate the shares of the value, the first N - 1 values are random, the last one is computed
    auto& value_shares = all_parties_shares.value_shares;
    std::generate_n(value_shares.begin(), N - 1, getRand<SemiShrType>);
    value_shares.back() = masked_value - std::accumulate(value_shares.begin(), value_shares.end() - 1,
                                                         SemiShrType(0)); // we can't use 0LL here

    // generate the shares of the mac in the same way
    auto& mac_shares = all_parties_shares.mac_shares;
    std::generate_n(mac_shares.begin(), N - 1, getRand<SemiShrType>);
    mac_shares.back() = mac - std::accumulate(mac_shares.begin(), mac_shares.end() - 1,
                                              SemiShrType(0));

    return all_parties_shares;
}


template <IsSpdz2kShare ShrType, std::size_t N>
typename FakeParty<ShrType, N>::AllPartiesSharesVec FakeParty<ShrType, N>::
GenerateAllPartiesShares(const std::vector<ClearType>& value) const {
    AllPartiesSharesVec all_parties_shares;
    auto& value_shares = all_parties_shares.value_shares;
    auto& mac_shares = all_parties_shares.mac_shares;

    auto size = value.size();

    std::ranges::for_each(value_shares, [size](auto& vec) { vec.resize(size); });
    std::ranges::for_each(mac_shares, [size](auto& vec) { vec.resize(size); });

    for (std::size_t vec_idx = 0; vec_idx < size; ++vec_idx) {
        auto shares_i = GenerateAllPartiesShares(value[vec_idx]);

        for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
            value_shares[party_idx][vec_idx] = shares_i.value_shares[party_idx];
            mac_shares[party_idx][vec_idx] = shares_i.mac_shares[party_idx];
        }
    }

    return all_parties_shares;
}


template <IsSpdz2kShare ShrType, std::size_t N>
void FakeParty<ShrType, N>::WriteSharesToAllParites(const std::array<std::vector<SemiShrType>, N>& shares,
                                                    const std::array<std::vector<SemiShrType>, N>& macs) {
    for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
        auto& output_file = ithPartyFile(party_idx);

        for (std::size_t vec_idx = 0; vec_idx < shares[party_idx].size(); ++vec_idx) {
            output_file << shares[party_idx][vec_idx] << ' ' << macs[party_idx][vec_idx] << '\n';
        }
    }
}

template <IsSpdz2kShare ShrType, std::size_t N>
void FakeParty<ShrType, N>::WriteClearToIthParty(const std::vector<ClearType>& values, std::size_t party_id) {
    auto& output_file = ithPartyFile(party_id);
    std::ranges::for_each(values, [&output_file](auto value) { output_file << value << '\n'; });
}


template <IsSpdz2kShare ShrType, std::size_t N>
void FakeParty<ShrType, N>::WriteClearToAllParties(const std::vector<ClearType>& values) {
    for (std::size_t party_idx = 0; party_idx < N; ++party_idx) {
        WriteClearToIthParty(values, party_idx);
    }
}


} // namespace md_ml

#endif //MD_ML_FAKEPARTY_H

// By Boshi Yuan

#ifndef MD_ML_FIXED_POINT_H
#define MD_ML_FIXED_POINT_H


#include <type_traits>
#include <vector>
#include <algorithm>


namespace md_ml {

template <typename ClearType>
ClearType double2fix(double x);

template <typename ClearType>
std::vector<ClearType> double2fixVec(const std::vector<double>& x);

template <typename ClearType>
double fix2double(ClearType x);

template <typename ClearType>
std::vector<double> fix2doubleVec(const std::vector<ClearType>& x);


namespace FixedPoint {
    const int fractionBits = 20;
    const int truncateValue = 1 << fractionBits;
}


template <typename ClearType>
ClearType double2fix(double x) {
    return static_cast<ClearType>(x * FixedPoint::truncateValue);
}

template <typename ClearType>
std::vector<ClearType> double2fixVec(const std::vector<double>& x) {
    std::vector<ClearType> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = double2fix<ClearType>(x[i]);
    }
    return res;
}


template <typename ClearType>
double fix2double(ClearType x) {
    return static_cast<double>(static_cast<std::make_signed_t<ClearType>>(x)) / FixedPoint::truncateValue;
}

template <typename ClearType>
std::vector<double> fix2doubleVec(const std::vector<ClearType>& x) {
    std::vector<double> res(x.size());
    std::transform(x.begin(), x.end(), res.begin(), fix2double<ClearType>);
    return res;
}

} // namespace md_ml

#endif //MD_ML_FIXED_POINT_H

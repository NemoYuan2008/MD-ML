// By Boshi Yuan

#ifndef MD_ML_LINEAR_ALGEBRA_H
#define MD_ML_LINEAR_ALGEBRA_H


#include <vector>
#include <algorithm>
#include <execution>
#include <concepts>

#include <Eigen/Core>

namespace md_ml {


template <typename T>
inline
std::vector<T> matrixAdd(const std::vector<T>& x, const std::vector<T>& y) {
    std::vector<T> output(x.size());

    // I found that std::transform is faster than Eigen's implementation of matrix addition,
    // but std::ranges::transform is slower, so I decided to use std::transform
    // Also, we can use std::execution::par_unseq to parallelize the computation
    // However, the parallelization is only available in libstdc++ (Linux), not available in libc++ (macOS),
    // so we detect it using macros to avoid compilation errors on macOS.
#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL   // or !__cpp_lib_execution?
    std::transform(x.begin(), x.end(), y.begin(), output.begin(), std::plus<T>());
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), output.begin(), std::plus<T>());
#endif

    return output;
}


template <typename T>
inline
void matrixAddAssign(std::vector<T>& x, const std::vector<T>& y) {
#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
    std::transform(x.begin(), x.end(), y.begin(), x.begin(), std::plus<T>());
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), x.begin(), std::plus<T>());
#endif
}


template <typename T1, std::integral T2>
inline
std::vector<T1> matrixAddConstant(const std::vector<T1>& x, T2 constant) {
    std::vector<T1> output(x.size());
#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
    std::transform(x.begin(), x.end(), output.begin(),
                   [constant](T1 val) { return val + constant; });
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), output.begin(),
                   [constant](T1 val) { return val + constant; });
#endif
    return output;
}


template <typename T>
inline
std::vector<T> matrixSubtract(const std::vector<T>& x, const std::vector<T>& y) {
    std::vector<T> output(x.size());

#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
    std::transform(x.begin(), x.end(), y.begin(), output.begin(), std::minus<T>());
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), output.begin(), std::minus<T>());
#endif

    return output;
}


template <typename T>
inline
void matrixSubtractAssign(std::vector<T>& x, const std::vector<T>& y) {
#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
    std::transform(x.begin(), x.end(), y.begin(), x.begin(), std::minus<T>());
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), x.begin(), std::minus<T>());
#endif
}


// matrix scalar product
template <typename T>
inline
std::vector<T> matrixScalar(const std::vector<T>& x, T scalar) {
    std::vector<T> output(x.size());
#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
    std::transform(x.begin(), x.end(), output.begin(), [scalar](T val) { return scalar * val; });
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), output.begin(),
                   [scalar](T val) { return scalar * val; });
#endif
    return output;
}

template <typename T>
inline
void matrixScalarAssign(std::vector<T>& x, T scalar) {
#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
    std::transform(x.begin(), x.end(), x.begin(), [scalar](T val) { return scalar * val; });
#else
    std::transform(std::execution::par_unseq, x.begin(), x.end(), x.begin(), [scalar](T val) { return scalar * val; });
#endif
}


template <typename T>
inline
void matrixMultiply(const T* lhs, const T* rhs, T* output,
                    std::size_t dim_row, std::size_t dim_mid, std::size_t dim_col) {
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<const MatrixType> matrix_lhs(lhs, dim_row, dim_mid);
    Eigen::Map<const MatrixType> matrix_rhs(rhs, dim_mid, dim_col);
    Eigen::Map<MatrixType> matrix_output(output, dim_row, dim_col);

    matrix_output = matrix_lhs * matrix_rhs;
}


template <typename T>
inline
std::vector<T> matrixMultiply(const std::vector<T>& lhs, const std::vector<T>& rhs,
                              std::size_t dim_row, std::size_t dim_mid, std::size_t dim_col) {
    std::vector<T> output(dim_row * dim_col);
    matrixMultiply(lhs.data(), rhs.data(), output.data(), dim_row, dim_mid, dim_col);
    return output;
}


} // namespace md_ml


#endif //MD_ML_LINEAR_ALGEBRA_H

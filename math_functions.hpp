#ifndef MATH_FUNCTIONS_HPP
#define MATH_FUNCTIONS_HPP

#include <cmath>
#include <cstring>
#include <mkl.h>

template <typename Dtype>
void caffe_set(const int n, const Dtype alpha, Dtype* y) {
  if (alpha == 0) {
    memset(y, 0, sizeof(Dtype) * n);
  } else {
    for (int i = 0; i < n; i++) {
      y[i] = alpha;
    }
  }
}
template void caffe_set<int>(const int n, const int a, int* y);
template void caffe_set<float>(const int n, const float a, float* y);
template void caffe_set<double>(const int n, const double a, double* y);

template <typename Dtype>
void caffe_axpy(const int n, const Dtype a, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_sincos(const int n, const Dtype* a, Dtype* y, Dtype* z);

template <typename Dtype>
void caffe_csrmv(const CBLAS_TRANSPOSE transa, const int m, const int k,
                 const Dtype alpha, const Dtype* val, const int* ro,
                 const int* ci, const Dtype* x, const Dtype beta, Dtype* y);

#endif

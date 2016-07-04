#include "math_functions.hpp"
#include <mkl.h>

template <>
void caffe_axpy(const int n, const float a, const float* x, float* y) {
  cblas_saxpy(n, a, x, 1, y, 1);
}

template <>
void caffe_axpy(const int n, const double a, const double* x, double* y) {
  cblas_daxpy(n, a, x, 1, y, 1);
}

template <>
void caffe_sincos(int n, const float* a, float* y, float* z) {
  vsSinCos(n, a, y, z);
}

template <>
void caffe_sincos(int n, const double* a, double* y, double* z) {
  vdSinCos(n, a, y, z);
}

template <>
void caffe_csrmv(const CBLAS_TRANSPOSE transa, const int m, const int k,
                 const float alpha, const float* val, const int* ro,
                 const int* ci, const float* x, const float beta, float* y) {
  const char* ta = transa == CblasTrans ? "T" : "N";
  const char* matdescra = "GUUC";
  mkl_scsrmv(ta, &m, &k, &alpha, matdescra, val, ci, ro, ro+1, x, &beta, y);
}

template <>
void caffe_csrmv(const CBLAS_TRANSPOSE transa, const int m, const int k,
                 const double alpha, const double* val, const int* ro,
                 const int* ci, const double* x, const double beta, double* y) {
  const char* ta = transa == CblasTrans ? "T" : "N";
  const char* matdescra = "GUUC";
  mkl_dcsrmv(ta, &m, &k, &alpha, matdescra, val, ci, ro, ro+1, x, &beta, y);
}


#include <cmath>
#include "common.hpp"
#include "math_functions.hpp"

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <typename Dtype>
__global__ void sincos_kernel(const int n, const Dtype* a,
                              Dtype* y, Dtype* z);

template <>
__global__ void sincos_kernel<float>(const int n, const float* a,
                                     float* y, float* z) {
  CUDA_KERNEL_LOOP(i, n) {
    sincosf(a[i], y+i, z+i);
  }
}

template <>
__global__ void sincos_kernel<double>(const int n, const double* a,
                                      double* y, double* z) {
  CUDA_KERNEL_LOOP(i, n) {
    sincos(a[i], y+i, z+i);
  }
}

template <typename Dtype>
void caffe_gpu_sincos(const int n, const Dtype* a, Dtype* y, Dtype* z) {
  sincos_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, a, y, z);
  CUDA_POST_KERNEL_CHECK;
}
template void caffe_gpu_sincos<float>(const int, const float*, float*, float*);
template void caffe_gpu_sincos<double>(const int, const double*, double*, double*);

template <>
void caffe_gpu_csrmv(const CBLAS_TRANSPOSE transa, const int m, const int n,
                     const int nnz, const float alpha, const float* val, const int* ro,
                     const int* ci, const float* x, const float beta, float* y) {
  cusparseOperation_t ta = transa == CblasTrans ?
      CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseMatDescr_t desc; cusparseCreateMatDescr(&desc);
  CUSPARSE_CHECK(cusparseScsrmv(Caffe::cusparse_handle(), ta, m, n, nnz,
                                &alpha, desc, val, ro, ci, x, &beta, y));
}

template <>
void caffe_gpu_csrmv(const CBLAS_TRANSPOSE transa, const int m, const int n,
                     const int nnz, const double alpha, const double* val, const int* ro,
                     const int* ci, const double* x, const double beta, double* y) {
  cusparseOperation_t ta = transa == CblasTrans ?
      CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseMatDescr_t desc; cusparseCreateMatDescr(&desc);
  CUSPARSE_CHECK(cusparseDcsrmv(Caffe::cusparse_handle(), ta, m, n, nnz,
                                &alpha, desc, val, ro, ci, x, &beta, y));
}


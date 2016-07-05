#ifndef HOUGH_BASIS_HPP
#define HOUGH_BASIS_HPP

#include <vector>
#include <cmath>
#include "math_functions.hpp"
#include "common.hpp"
#include "syncedmem.hpp"


template<typename Dtype>
class HoughBasis {
public:
  HoughBasis(int H, int W, int THETA=-1, int RHO=-1) :
      H_(H), W_(W) {
    theta_min_ = -90;
    theta_max_ = 90;
    rho_min_ = -int(std::floor(std::sqrt(H*H+W*W)));
    rho_max_ = int(std::ceil(std::sqrt(H*H+W*W)));

    THETA = THETA > 0 ? THETA : (theta_max_-theta_min_);
    RHO = RHO > 0 ? RHO : (rho_max_-rho_min_);
    this->THETA_ = THETA;
    this->RHO_ = RHO;

    theta_step_ = Dtype(theta_max_-theta_min_) / THETA;
    rho_step_ = Dtype(rho_max_-rho_min_) / RHO;

    v_.reset(new SyncedMemory(sizeof(Dtype) * H*W*THETA));
    ci_.reset(new SyncedMemory(sizeof(int) * H*W*THETA));
    ro_.reset(new SyncedMemory(sizeof(int) * (1+H*W)));

    if (Caffe::mode() == Caffe::CPU)
      Init_cpu();
    else
      Init_gpu();
  }

  void Init_gpu();

  void Init_cpu() {
    const Dtype pi = std::acos(-1);
    SyncedMemory theta_(sizeof(Dtype) * THETA_);
    for (int theta_i = 0; theta_i < THETA_; theta_i++) {
      Dtype theta = theta_min_ + theta_i * theta_step_;
      ((Dtype*) theta_.mutable_cpu_data())[theta_i] = theta * pi / 180;
    }
    SyncedMemory sin_(sizeof(Dtype) * THETA_);
    SyncedMemory cos_(sizeof(Dtype) * THETA_);
    caffe_sincos(THETA_, (const Dtype*) theta_.cpu_data(),
                 (Dtype*) sin_.mutable_cpu_data(),
                 (Dtype*) cos_.mutable_cpu_data());

    // TODO: parallel for
    for (int idx = 0; idx < H_*W_*THETA_; idx++) {
      const int hw = idx / THETA_;
      const int theta_i = idx % THETA_;
      const int h = hw / W_;
      const int w = hw % W_;
      const int ro = hw * THETA_;

      Dtype rho = h * ((Dtype*) sin_.cpu_data())[theta_i] +
                  w * ((Dtype*) cos_.cpu_data())[theta_i];
      int rho_i = int( (rho-rho_min_)/rho_step_ );
      int ci = theta_i * RHO_ + rho_i;  // col idx
      val_mutable_cpu_data()[ro+theta_i] = Dtype(1);
      ci_mutable_cpu_data()[ro+theta_i] = ci;

      if (theta_i == 0) {
        ro_mutable_cpu_data()[hw] = ro;
        if (idx == H_*W_*THETA_-1) {
          ro_mutable_cpu_data()[hw+1] = ro + THETA_;
        }
      }
    }
  }

  inline int H() { return H_; }
  inline int W() { return W_; }
  inline int RHO() { return RHO_; }
  inline int THETA() { return THETA_; }
  inline const Dtype* val_cpu_data() { return (const Dtype*) v_->cpu_data(); }
  inline const int* ci_cpu_data() { return (const int*) ci_->cpu_data(); }
  inline const int* ro_cpu_data() { return (const int*) ro_->cpu_data(); }
  inline const int* pb_cpu_data() { return (const int*) ro_->cpu_data(); }
  inline const int* pe_cpu_data() { return (const int*) ro_->cpu_data()+1; }
  inline const Dtype* val_gpu_data() { return (const Dtype*) v_->gpu_data(); }
  inline const int* ci_gpu_data() { return (const int*) ci_->gpu_data(); }
  inline const int* ro_gpu_data() { return (const int*) ro_->gpu_data(); }
  inline const int* pb_gpu_data() { return (const int*) ro_->gpu_data(); }
  inline const int* pe_gpu_data() { return (const int*) ro_->gpu_data()+1; }
  inline int theta_min() { return theta_min_; }
  inline int theta_max() { return theta_max_; }
  inline int rho_min() { return rho_min_; }
  inline int rho_max() { return rho_max_; }
  inline int nnz() { return H_*W_*THETA_; }

protected:
  inline Dtype* val_mutable_cpu_data() { return (Dtype*) v_->mutable_cpu_data(); }
  inline int* ci_mutable_cpu_data() { return (int*) ci_->mutable_cpu_data(); }
  inline int* ro_mutable_cpu_data() { return (int*) ro_->mutable_cpu_data(); }
  inline int* pb_mutable_cpu_data() { return (int*) ro_->mutable_cpu_data(); }
  inline int* pe_mutable_cpu_data() { return (int*) ro_->mutable_cpu_data()+1; }
  inline Dtype* val_mutable_gpu_data() { return (Dtype*) v_->mutable_gpu_data(); }
  inline int* ci_mutable_gpu_data() { return (int*) ci_->mutable_gpu_data(); }
  inline int* ro_mutable_gpu_data() { return (int*) ro_->mutable_gpu_data(); }
  inline int* pb_mutable_gpu_data() { return (int*) ro_->mutable_gpu_data(); }
  inline int* pe_mutable_gpu_data() { return (int*) ro_->mutable_gpu_data()+1; }

private:
  int H_;            // range of height in spatial domain
  int W_;            // rnage of width in spatial domain
  int RHO_;          // rnage of rho in Hough domain
  int THETA_;        // range of theta in Hough domain
  int theta_min_;    // min of theta
  int theta_max_;    // max of theta
  int theta_step_;   // step of theta
  int rho_min_;      // min of rho
  int rho_max_;      // max of rho
  int rho_step_;     // step of rho

  shared_ptr<SyncedMemory> v_;   // values, array of Dtype
  shared_ptr<SyncedMemory> ci_;  // column indices, array of int
  shared_ptr<SyncedMemory> ro_;  // row offsets, array of int
};

#endif

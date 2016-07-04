#ifndef HOUGH_BASIS_HPP
#define HOUGH_BASIS_HPP

#include <vector>
#include <cmath>
#include "math_functions.hpp"

using std::vector;

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

    v_.resize(H*W*THETA);
    ci_.resize(H*W*THETA);
    ro_.resize(1+H*W);

    Dtype theta_step = Dtype(theta_max_-theta_min_) / THETA;
    Dtype rho_step = Dtype(rho_max_-rho_min_) / RHO;

    const Dtype pi = std::acos(-1);
    Dtype theta_[THETA];
    for (int theta_i = 0; theta_i < THETA; theta_i++) {
      Dtype theta = theta_min_ + theta_i * theta_step;
      theta_[theta_i] = theta * pi / 180;
    }
    Dtype sin_[THETA], cos_[THETA];
    caffe_sincos(THETA, (Dtype*) theta_, (Dtype*) sin_, (Dtype*) cos_);

    ro_[0] = 0;
    // TODO: parallel for
    for (int hw = 0; hw < H*W; hw++) {
      const int h = hw / W;
      const int w = hw % W;
      const int ro = hw * THETA;  // row offset

      Dtype rho_[THETA];
      caffe_set(THETA, Dtype(0), (Dtype*) rho_);
      caffe_axpy(THETA, Dtype(h), sin_, rho_);
      caffe_axpy(THETA, Dtype(w), cos_, rho_);
      for (int theta_i = 0; theta_i < THETA; theta_i++) {
        int rho_i = int((rho_[theta_i]-rho_min_)/rho_step);
        int ci = theta_i * RHO + rho_i;  // col idx
        v_[ro+theta_i] = Dtype(1);
        ci_[ro+theta_i] = ci;
      }

      ro_[hw+1] = ro;
    }
  }

  inline int H() { return H_; }
  inline int W() { return W_; }
  inline int RHO() { return RHO_; }
  inline int THETA() { return THETA_; }
  inline Dtype* val() { return v_.data(); }
  inline int* ci() { return ci_.data(); }
  inline int* ro() { return ro_.data(); }
  inline int* pb() { return ro_.data(); }
  inline int* pe() { return ro_.data()+1; }
  inline int theta_min() { return theta_min_; }
  inline int theta_max() { return theta_max_; }
  inline int rho_min() { return rho_min_; }
  inline int rho_max() { return rho_max_; }

private:
  int H_;            // range of height in spatial domain
  int W_;            // rnage of width in spatial domain
  int RHO_;          // rnage of rho in Hough domain
  int THETA_;        // range of theta in Hough domain
  int theta_min_;    // min of theta
  int theta_max_;    // max of theta
  int rho_min_;      // min of rho
  int rho_max_;      // max of rho

  vector<Dtype> v_;  // values
  vector<int> ci_;   // column indices
  vector<int> ro_;   // row offsets
};

#endif

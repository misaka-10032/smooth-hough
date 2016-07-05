#include <cassert>
#include <iostream>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mkl.h>
#include "common.hpp"
#include "hough_basis.hpp"
#include "math_functions.hpp"
#include "cycle_timer.h"

using std::cout;
using std::endl;
using namespace cv;

void saveppm(char* fname, Mat mat) {
  double min, max;
  minMaxLoc(mat, &min, &max);
  Mat normalized = (mat-min) / (max-min) * 255;
  Mat saved; normalized.convertTo(saved, CV_8U);
  imwrite(fname, saved);
}

void forward_cpu(HoughBasis<float>& hb, const float* x, float* y) {
  caffe_csrmv(CblasTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(), float(1),
              hb.val_cpu_data(), hb.ro_cpu_data(), hb.ci_cpu_data(),
              x, float(0), y);
}


void backward_cpu(HoughBasis<float>& hb, const float* dy, float* dx) {
  caffe_csrmv(CblasNoTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(), float(1),
              hb.val_cpu_data(), hb.ro_cpu_data(), hb.ci_cpu_data(),
              dy, float(0), dx);
}

void forward_gpu(HoughBasis<float>& hb, const float* x, float* y) {
  caffe_gpu_csrmv(CblasTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(),
                  hb.nnz(), float(1), hb.val_cpu_data(),
                  hb.ro_cpu_data(), hb.ci_cpu_data(), x, float(0), y);
}

void backward_gpu(HoughBasis<float>& hb, const float* dy, float* dx) {
  caffe_gpu_csrmv(CblasNoTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(),
                  hb.nnz(), float(1), hb.val_cpu_data(),
                  hb.ro_cpu_data(), hb.ci_cpu_data(), dy, float(0), dx);
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 7) {
    cout << "Usage: " << argv[0] << "<cpu/gpu> <in_prob> <in_tgt> "
         "<out_hf_prob> <out_hf_tgt> <out_grad>" << endl;
    return 1;
  }
  double tic, toc;

  if ( strcmp(argv[1], "cpu") == 0 ) {
    Caffe::set_mode(Caffe::CPU);
  } else if ( strcmp(argv[1], "gpu") == 0 ) {
    Caffe::set_mode(Caffe::GPU);
  } else {
    cout << "argv[1] should be either cpu or gpu." << endl;
    return 1;
  }

  // Read images
  Mat img = imread(argv[2], 0);
  SyncedMemory prob_sm(sizeof(float)*img.rows*img.cols);
  Mat prob(img.rows, img.cols, CV_32F, prob_sm.mutable_cpu_data());
  img.convertTo(prob, CV_32F);
  prob /= 255;
  assert(prob.isContinuous());

  Mat tgt_raw = imread(argv[3], 0);
  SyncedMemory tgt_sm(sizeof(float)*tgt_raw.rows*tgt_raw.cols);
  Mat tgt(tgt_raw.rows, tgt_raw.cols, CV_32F, tgt_sm.mutable_cpu_data());
  tgt_raw.convertTo(tgt, CV_32F);
  tgt /= 255;
  assert(tgt.isContinuous());

  // Set up baisis
  tic = CycleTimer::currentSeconds();
  HoughBasis<float> hb(prob.rows, prob.cols);
  toc = CycleTimer::currentSeconds();
  cout << "hough basis takes " << toc-tic << " seconds" << endl;

  // Forward
  SyncedMemory hft_prob_sm(sizeof(float)*hb.THETA()*hb.RHO());
  SyncedMemory hft_tgt_sm(sizeof(float)*hb.THETA()*hb.RHO());
  if (Caffe::mode() == Caffe::CPU) {
    const float* prob_cpu_data = (const float*) prob_sm.cpu_data();
    const float* tgt_cpu_data = (const float*) tgt_sm.cpu_data();
    float* hft_prob_mutable_cpu_data = (float*) hft_prob_sm.mutable_cpu_data();
    float* hft_tgt_mutable_cpu_data = (float*) hft_tgt_sm.mutable_cpu_data();
    tic = CycleTimer::currentSeconds();
    forward_cpu(hb, prob_cpu_data, hft_prob_mutable_cpu_data);
    forward_cpu(hb, tgt_cpu_data, hft_tgt_mutable_cpu_data);
    toc = CycleTimer::currentSeconds();
  } else {
    const float* prob_gpu_data = (const float*) prob_sm.gpu_data();
    const float* tgt_gpu_data = (const float*) tgt_sm.gpu_data();
    float* hft_prob_mutable_gpu_data = (float*) hft_prob_sm.mutable_gpu_data();
    float* hft_tgt_mutable_gpu_data = (float*) hft_tgt_sm.mutable_gpu_data();
    cudaThreadSynchronize();
    tic = CycleTimer::currentSeconds();
    forward_gpu(hb, prob_gpu_data, hft_prob_mutable_gpu_data);
    forward_gpu(hb, tgt_gpu_data, hft_tgt_mutable_gpu_data);
    cudaThreadSynchronize();
    toc = CycleTimer::currentSeconds();
  }
  cout << "hough forward takes " << toc-tic << " seconds" << endl;
  Mat hft_prob(hb.THETA(), hb.RHO(), CV_32F, hft_prob_sm.mutable_cpu_data());
  Mat hft_tgt(hb.THETA(), hb.RHO(), CV_32F, hft_tgt_sm.mutable_cpu_data());
  assert(hft_prob.isContinuous());
  assert(hft_tgt.isContinuous());

  Mat hf_prob(hb.RHO(), hb.THETA(), CV_32F);
  Mat hf_tgt(hb.RHO(), hb.THETA(), CV_32F);
  transpose(hft_prob, hf_prob);
  transpose(hft_tgt, hf_tgt);
  assert(hft_prob.isContinuous());
  assert(hft_tgt.isContinuous());
  saveppm(argv[4], hf_prob);
  saveppm(argv[5], hf_tgt);

  // Backward
  SyncedMemory hft_diff_sm(sizeof(float)*hb.THETA()*hb.RHO());
  SyncedMemory prob_diff_sm(sizeof(float)*prob.rows*prob.cols);
  Mat hft_diff(hb.THETA(), hb.RHO(), CV_32F, hft_diff_sm.mutable_cpu_data());
  hft_diff = hft_prob - hft_tgt;
  if (Caffe::mode() == Caffe::CPU) {
    const float* hft_diff_cpu_data = (const float*) hft_diff_sm.cpu_data();
    float* prob_diff_mutable_cpu_data = (float*) prob_diff_sm.mutable_cpu_data();
    tic = CycleTimer::currentSeconds();
    backward_cpu(hb, hft_diff_cpu_data, prob_diff_mutable_cpu_data);
    toc = CycleTimer::currentSeconds();
  } else {
    const float* hft_diff_gpu_data = (const float*) hft_diff_sm.gpu_data();
    float* prob_diff_mutable_gpu_data = (float*) prob_diff_sm.mutable_gpu_data();
    cudaThreadSynchronize();
    tic = CycleTimer::currentSeconds();
    backward_gpu(hb, hft_diff_gpu_data, prob_diff_mutable_gpu_data);
    cudaThreadSynchronize();
    toc = CycleTimer::currentSeconds();
  }
  cout << "hough backward takes " << toc-tic << " seconds" << endl;
  Mat prob_diff(prob.rows, prob.cols, CV_32F, prob_diff_sm.mutable_cpu_data());
  saveppm(argv[6], prob_diff);

  return 0;
}


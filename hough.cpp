#include <cassert>
#include <iostream>
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

int main(int argc, char** argv) {
  if (argc != 6) {
    cout << "Usage: " << argv[0] << " <in_prob> <in_tgt> "
         "<out_hf_prob> <out_hf_tgt> <out_grad>" << endl;
    return 1;
  }
  double tic, toc;
  Caffe::set_mode(Caffe::CPU);

  // Read images
  Mat img = imread(argv[1], 0);
  Mat prob; img.convertTo(prob, CV_32F);
  prob /= 255;
  assert(prob.isContinuous());

  Mat tgt_raw = imread(argv[2], 0);
  Mat tgt; tgt_raw.convertTo(tgt, CV_32F);
  tgt /= 255;
  assert(tgt.isContinuous());

  // Set up baisis
  tic = CycleTimer::currentSeconds();
  HoughBasis<float> hb(prob.rows, prob.cols);
  toc = CycleTimer::currentSeconds();
  cout << "hough basis takes " << toc-tic << " seconds" << endl;

  // Forward
  Mat hft_prob(hb.THETA(), hb.RHO(), CV_32F);
  Mat hft_tgt(hb.THETA(), hb.RHO(), CV_32F);
  assert(hft_prob.isContinuous());
  assert(hft_tgt.isContinuous());

  tic = CycleTimer::currentSeconds();
  caffe_csrmv(CblasTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(), float(1),
              hb.val_cpu_data(), hb.ro_cpu_data(), hb.ci_cpu_data(),
              (float*) prob.ptr(), float(0), (float*) hft_prob.ptr());
  caffe_csrmv(CblasTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(), float(1),
              hb.val_cpu_data(), hb.ro_cpu_data(), hb.ci_cpu_data(),
              (float*) tgt.ptr(), float(0), (float*) hft_tgt.ptr());
  toc = CycleTimer::currentSeconds();
  cout << "hough forward takes " << toc-tic << " seconds" << endl;

  Mat hf_prob(hb.RHO(), hb.THETA(), CV_32F);
  Mat hf_tgt(hb.RHO(), hb.THETA(), CV_32F);
  transpose(hft_prob, hf_prob);
  transpose(hft_tgt, hf_tgt);
  assert(hft_prob.isContinuous());
  assert(hft_tgt.isContinuous());
  saveppm(argv[3], hf_prob);
  saveppm(argv[4], hf_tgt);

  // Backward
  Mat hft_diff = hft_prob - hft_tgt;
  assert(hft_diff.depth() == CV_32F);
  Mat prob_diff(prob.rows, prob.cols, CV_32F);
  caffe_csrmv(CblasNoTrans, hb.H()*hb.W(), hb.RHO()*hb.THETA(), float(1),
              hb.val_cpu_data(), hb.ro_cpu_data(), hb.ci_cpu_data(),
              (float*) hft_diff.ptr(), float(0), (float*) prob_diff.ptr());
  toc = CycleTimer::currentSeconds();
  cout << "hough backward takes " << toc-tic << " seconds" << endl;
  saveppm(argv[5], prob_diff);

  return 0;
}

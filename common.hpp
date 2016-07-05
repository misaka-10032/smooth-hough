#ifndef COMMON_HPP
#define COMMON_HPP

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include "device_alternate.hpp"

using boost::shared_ptr;
using std::vector;

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

class Caffe {
public:
  enum Brew { CPU, GPU };
  static Caffe& Get();

  inline static Brew mode() { return Get().mode_; }
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  inline static cusparseHandle_t cusparse_handle() {
    return Get().cusparse_handle_;
  }
private:
  Caffe();
  Brew mode_;
  cublasHandle_t cublas_handle_;
  cusparseHandle_t cusparse_handle_;
};

#endif

CXX := g++
CXXFLAGS :=
CXXFLAGS += -g
LDFLAGS  :=
LDLIBS   :=
# opencv
CXXFLAGS += -I/home/longqic/local/include
LDFLAGS  += -L/home/longqic/local/lib
LDLIBS   += -lopencv_core -lopencv_highgui
# mkl
CXXFLAGS += -I/opt/intel/mkl/include
LDFLAGS  += -L/opt/intel/mkl/lib/intel64
LDLIBS   += -lmkl_rt
# boost
LDLIBS   += -lboost_thread -lboost_system
# cuda
CXXFLAGS += -I/usr/local/cuda/include
LDFLAGS  += -L/usr/local/cuda/lib64
LDLIBS   += -lcudart
# glog
CXXFLAGS += -I/home/longqic/local/opt/glog-0.3.3/install/include
LDFLAGS  += -L/home/longqic/local/opt/glog-0.3.3/install/lib
LDLIBS   += -lglog

# nvcc
NVCC      := nvcc
NVCCFLAGS :=
NVCCFLAGS += -G
NVCCFLAGS += -gencode arch=compute_20,code=sm_20 \
             -gencode arch=compute_20,code=sm_21 \
             -gencode arch=compute_30,code=sm_30 \
             -gencode arch=compute_35,code=sm_35 \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_50,code=compute_50

# dep
C_HDRS   := $(shell find . -name "*.h")
CXX_HDRS := $(shell find . -name "*.hpp")
CXX_SRCS := $(shell find . -name "*.cpp")
CU_SRCS  := $(shell find . -name "*.cu")
DEP      := $(C_HDRS) $(CXX_HDRS) $(CXX_SRCS) $(CU_SRCS)

BUILD := build
OBJS  :=
OBJS  += $(patsubst %.cpp,$(BUILD)/%.o,$(CXX_SRCS))
OBJS  += $(patsubst %.cu,$(BUILD)/%.cu.o,$(CU_SRCS))

#########
# rules #
#########

.PHONY: pre clean

all: hough

pre:
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(BUILD)/%.cu.o: %.cu
	$(NVCC) $< $(CXXFLAGS) $(NVCCFLAGS) -c -o $@

%.cpp: %.hpp

%.cu: %.hpp

hough: pre $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(BUILD)/$@ $(OBJS) $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD)


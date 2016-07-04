CC := g++
CFLAGS :=
CFLAGS += -g
CFLAGS += -I/home/longqic/local/include -L/home/longqic/local/lib
CFLAGS += -lopencv_core -lopencv_highgui
CFLAGS += -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64
CFLAGS += -lmkl_rt

all: hough

hough: hough.cpp hough_basis.hpp math_functions.cpp
	$(CC) $(CFLAGS) $^ -o $@

%.cpp: %.hpp

clean:
	rm -f hough

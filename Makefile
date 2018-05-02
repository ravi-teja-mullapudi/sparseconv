ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

LIB_PATH := -L$(ROOT_DIR)/third_party/FALCON/build -L$(ROOT_DIR)/third_party/memkind/.libs
LIBS := -lfalcon -lmemkind

conv_naive: conv_naive.cpp
	icpc -O3 -march=native -fno-alias -ipo -std=c++11 -qopenmp -qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout -xhost -restrict $(LIB_PATH) conv_naive.cpp -o conv_naive $(LIBS)
	#g++ -O3 -march=native -std=c++11 -fopenmp conv_naive.cpp -o conv_naive $(LIB_PATH) $(LIBS)

clean:
	rm -rf conv_naive

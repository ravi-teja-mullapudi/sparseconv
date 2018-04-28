conv_naive: conv_naive.cpp
	g++ -O3 -std=c++11 -fopenmp conv_naive.cpp -o conv_naive

clean:
	rm -rf conv_naive

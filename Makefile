conv_naive: conv_naive.cpp
	icpc -O3 -march=native -fno-alias -ipo -std=c++11 -qopenmp -qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout conv_naive.cpp -o conv_naive
	#g++ -O3 -march=native -std=c++11 -fopenmp conv_naive.cpp -o conv_naive

clean:
	rm -rf conv_naive

#!/bash/bin

CXX = g++
CXXFLAGS = -std=c++11 -O2 -lm  -lgsl -lgslcblas -fopenmp


prog: main.o lib.o
	$(CXX)  -o $@  $^ $(CXXFLAGS) 

main.o : main.cpp lib.hpp
	$(CXX) -c main.cpp  $(CXXFLAGS) 

lib.o: lib.cpp lib.hpp
	$(CXX) -c lib.cpp $(CXXFLAGS) 


.PHONEY: clean
clean:
	rm -rf  *.pyc 
	rm -rf ../data/*.txt 
	rm -rf ../data/text/*
	rm -rf ../data/fig/* 
	rm -rf ../data/npz/*
	rm -f *.png *.out  

.PHONEY: eradicate
eradicate: clean
	rm -rf *.o  prog 
	# rm -rf .vscode
	rm -rf __pycache__/
	rm -rf out.txt

.PHONY: clean, eradicate

CXX = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp
SERIAL_CXXFLAGS = -O3 -std=c++17
TARGET = run_fastSKG_omp
SRC = fastSKG_omp.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) || $(CXX) $(SERIAL_CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

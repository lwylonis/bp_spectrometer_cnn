INC := -I ./include 
INC_XCL := 
#-I /opt/xilinx/xrt/include/
GXX_FLAGS := -w -O2 -std=c++17
LIB := -ltapa -lfrt -lglog -lgflags -lOpenCL
SRC := ./src

.DEFAULT_GOAL := cnn

cnn.o: $(SRC)/cnn.cpp
	tapa g++ -- $(GXX_FLAGS) -c $^ $(INC) $(INC_XCL)

main.o: $(SRC)/main.cpp
	tapa g++ -- $(GXX_FLAGS) -c $^ $(INC) $(INC_XCL)

host.o: $(SRC)/host.cpp
	tapa g++ -- $(GXX_FLAGS) -c $^ $(INC) $(INC_XCL)

cnn: cnn.o main.o host.o
	tapa g++ -- $(GXX_FLAGS) -o $@ $^ $(INC) $(INC_XCL) $(LIB)

swsim: cnn
	./cnn ./data

clean:
	rm *.o cnn

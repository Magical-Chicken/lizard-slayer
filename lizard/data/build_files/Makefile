CXX = g++
NVCC = nvcc
NVFLAGS = -Xcompiler '-fPIC'

HEADER = program.h
NVOBJ = kernel.o

CUDA_L64 =

TARGET_SO = user_program_cuda.so

all: $(TARGET_SO)
	@:

$(TARGET_SO): $(NVOBJ)
	$(NVCC) $(NVFLAGS) -dlink -o link.o $^
	$(CXX) -shared -o $@ $^ link.o $(CUDA_L64) -l cudart

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) -dc -o $@ $<

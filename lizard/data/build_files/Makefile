CXX = g++
NVCC = nvcc
NVFLAGS = -Xcompiler '-fPIC'

HEADER = program.h
NVOBJ = kernel.o

CUDA_L64 =
COMPUTE_LEVEL_FLAG =

TARGET_SO = user_program_cuda.so

all: $(TARGET_SO)
	@:

$(TARGET_SO): $(NVOBJ)
	$(NVCC) $(NVFLAGS) $(COMPUTE_LEVEL_FLAG) -dlink -o link.o $^
	$(CXX) -shared -o $@ $^ link.o $(CUDA_L64) -l cudart

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVFLAGS) $(COMPUTE_LEVEL_FLAG) -dc -o $@ $<

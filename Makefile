DEPS = cuda.h

all: cuda 
	 setup.py build_ext --build-lib lizard/

cuda: cuda.cu $(DEPS)
	nvcc -shared -o libcuda.so $< --compiler-options '-fPIC'

check:
	@tox

clean: 
	rm -rf build/
	rm -rf lizard/cuda.cpython-35m-x86_64-linux-gnu.so
	rm -rf libcuda.so

.PHONY: tox cuda

nvcc -shared -o libcuda.so cuda.cu -O3 -arch=sm_60 --compiler-options '-fPIC'
python3 setup.py build_ext --build-lib lizard/
export LD_LIBRARY_PATH=.

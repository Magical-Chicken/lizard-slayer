nvcc -shared -o libcuda.so cuda.cu --compiler-options '-fPIC'
python3 setup.py build_ext --build-lib lizard/
export LD_LIBRARY_PATH=.

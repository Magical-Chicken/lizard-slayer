check:
	@tox

clean: 
	rm -rf build/
	rm -rf lizard/cuda.cpython-35m-x86_64-linux-gnu.so
	rm -rf libcuda.so

.PHONY: tox cuda

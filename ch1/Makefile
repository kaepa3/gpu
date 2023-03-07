hello:hello.cu
	nvcc $< -o hello

run:hello.cu
	nvcc $< -o hello
	hello

sm:hello.cu
	nvcc -arch sm_20 $< -o hello

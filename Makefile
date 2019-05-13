all:
	gcc -Wall -std=c99 lenet_sequential.c main_sequential.c -lm -o sequential
	gcc -Wall -std=c99 -fopenmp -mavx lenet.c main.c -lm -o lenet
	gcc -Wall -std=c99 -fopenmp lenet_benchmark.c main.c -lm -o benchmark
	#nvcc -g -pg -I. --ptxas-options=-v -arch=sm_61 -o cuda_mnist.o -c mnist.cu
	#nvcc -g -pg -I. --ptxas-options=-v -arch=sm_61 -o cuda_mnist_file.o -c mnist_file.cu
	#nvcc -g -pg -I. --ptxas-options=-v -arch=sm_61 -o cuda_nn.o -c neural_network.cu
	#nvcc -g -pg -o cuda_mnist cuda_mnist.o cuda_mnist_file.o cuda_nn.o


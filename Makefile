all:
	gcc -Wall -std=c99 lenet_sequential.c main_sequential.c -lm -o sequential
	gcc -Wall -std=c99 -fopenmp lenet.c main.c -lm -o lenet
	gcc -Wall -std=c99 -fopenmp -O3 -mavx lenet.c main.c -lm -o o3
	gcc -Wall -std=c99 -fopenmp lenet_benchmark.c main.c -lm -o benchmark


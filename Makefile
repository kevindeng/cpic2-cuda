

CC = nvcc

CCOPTS = -O0 -arch=sm_20

all : cpic2

cpic2 : cpic2.o cpush2.o dtimer.o cpush2cuda.o
	$(CC) $(CCOPTS) -lm -o cpic2 cpic2.o cpush2.o dtimer.o cpush2cuda.o


# Compilation rules

dtimer.o : dtimer.c
	$(CC) $(CCOPTS) -c dtimer.c

cpush2.o : push2.c
	$(CC) $(CCOPTS) -o cpush2.o -c push2.c

cpic2.o : pic2.c
	$(CC) $(CCOPTS) -o cpic2.o -c pic2.c

cpush2cuda.o : push2cuda.cu
	$(CC) $(CCOPTS) -o cpush2cuda.o -c push2cuda.cu

clean :
	rm -f *.o *.mod

clobber: clean
	rm -f fpic2 cpic2 fpic2_c cpic2_f

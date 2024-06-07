OBJS = main.o linear.o nonlinear.o read.o conservation_laws.o flux.o spectrum.o initialise_forcing.o 

#this is for linux machines
CC = g++ -Wall -O3 -std=c++14 -fopenmp
CUFFTW_INC_PATH = -I/usr/local/cuda-11.7/targets/x86_64-linux/include
CUFFTW_LIB_PATH = -L/usr/local/cuda-11.7/targets/x86_64-linux/lib

DEBUG =
CFLAGS = -c $(DEBUG)
LFLAGS = $(DEBUG) -lcufftw -lcufft -lopenblas -lm -lpthread -llapack -larmadillo $(CUFFTW_INC_PATH) $(CUFFTW_LIB_PATH)


#the following is for macports in mac osx
#CC = g++-mp-10 -Wall -O3 -std=c++14 -fopenmp
#DEBUG = 
#CFLAGS = -c $(DEBUG) -I/opt/local/include
#LFLAGS =  $(DEBUG) -L/opt/local/lib -framework Accelerate -march=native -lfftw3 -lfftw3_threads -lopenblas -lm -lpthread -larmadillo

.PHONY:	clean

turb2d: $(OBJS)
	$(CC) $(OBJS) $(LFLAGS) -o turb2d

main.o: main.cpp const.h functions.h
	$(CC) $(CFLAGS) $(LFLAGS)  main.cpp
	
nonlinear.o: nonlinear.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) nonlinear.cpp

linear.o: linear.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) linear.cpp

read.o: read.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) read.cpp
    
conservation_laws.o: conservation_laws.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) conservation_laws.cpp

flux.o: flux.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) flux.cpp

spectrum.o: spectrum.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) spectrum.cpp

#Forcing.o: Forcing.cpp Const.h
#	$(CC) $(CFLAGS) Forcing.cpp

initialise_forcing.o: initialise_forcing.cpp const.h
	$(CC) $(CFLAGS) $(LFLAGS) initialise_forcing.cpp

clean:
	rm -rfv $(OBJS) *.o ./output/ ./turb2d


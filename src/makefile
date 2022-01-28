OBJS = main.o linear.o nonlinear.o read.o conservation_laws.o flux.o spectrum.o initialise_forcing.o 

#this is for linux machines
CC = g++ -Wall -O3 -std=c++14 -fopenmp
CFLAGS = -c -I/usr/local/include
LFLAGS = $(DEBUG) -lfftw3 -lfftw3_threads -lopenblas -L/usr/local/lib -lm -lpthread -llapack -larmadillo

#this is for aston math cluster
#CC = g++ -Wall -O3 -std=c++14 -fopenmp
#CFLAGS = -c -I/apps/lib64/openblas/include -I/home/lauriej/libs/include -I/home/lauriej/libs/usr/local/include
#LFLAGS = -lfftw3 -lfftw3_threads -L/apps/lib64/openblas/lib -L/home/lauriej/libs/lib -L/home/lauriej/libs/usr/local/lib64 -lm -lpthread -lopenblas -llapack -larmadillo


#the following is for macports in mac osx
CC = g++-mp-10 -Wall -O3 -std=c++14 -fopenmp
DEBUG = 
CFLAGS = -c $(DEBUG) -I/opt/local/include
LFLAGS =  $(DEBUG) -L/opt/local/lib -framework Accelerate -march=native -lfftw3 -lfftw3_threads -lopenblas -lm -lpthread -larmadillo

.PHONY:	clean

turb2d: $(OBJS)
	$(CC) $(OBJS) $(LFLAGS) -o turb2d

main.o: main.cpp const.h functions.h
	$(CC) $(CFLAGS) main.cpp
	
nonlinear.o: nonlinear.cpp const.h
	$(CC) $(CFLAGS) nonlinear.cpp

linear.o: linear.cpp const.h
	$(CC) $(CFLAGS) linear.cpp

read.o: read.cpp const.h
	$(CC) $(CFLAGS) read.cpp
    
conservation_laws.o: conservation_laws.cpp const.h
	$(CC) $(CFLAGS) conservation_laws.cpp

flux.o: flux.cpp const.h
	$(CC) $(CFLAGS) flux.cpp

spectrum.o: spectrum.cpp const.h
	$(CC) $(CFLAGS) spectrum.cpp

#Forcing.o: Forcing.cpp Const.h
#	$(CC) $(CFLAGS) Forcing.cpp

initialise_forcing.o: initialise_forcing.cpp const.h
	$(CC) $(CFLAGS) initialise_forcing.cpp

clean:
	rm -rfv $(OBJS) *.o ./output/ ./turb2d


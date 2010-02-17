#Path definitions:
I_PATH_X11 = /usr/X11R6/include
L_PATH_X11 = /usr/X11R6/lib
I_PATH_LOC = /usr/local/include/
L_PATH_LOC = /usr/local/lib/
I_PATH_PG = /opt/local/include/
L_PATH_PG = /opt/local/lib/
LIBS_FITS = -lcfitsio
LIBS_OPENCL = 

#ifdef __APPLE__
LIBS_FITS = ${LIBS_FITS} -lSystemStubs -lX11 
#else
LIBS_FITS = -lcfitsio
#endif

FFLAGS = 
CFLAGS = -O3

CC = gcc -Wall -O3

#DEFAULT

default: gpair


#ALL

all: opt

gpair.o: gpair.c
	$(CC) $(CFLAGS) -c gpair.c  -I$(I_PATH_LOC) -I$(I_PATH_PG)  -I$(I_PATH_X11)

read_fits.o: read_fits.c
	$(CC) $(CFLAGS) -c read_fits.c -I$(I_PATH_LOC)  -I$(I_PATH_X11) -I$(I_PATH_PG)

getoifits.o: getoifits.c
	$(CC) $(CFLAGS) -c getoifits.c -I$(I_PATH_LOC)  -I$(I_PATH_X11) -I$(I_PATH_PG)

#CLEANER

clean:
	rm -f *.o

#OPT PROGRAM

gpair: gpair.o read_fits.o getoifits.o
	$(CC) $(CFLAGS) -o gpair gpair.o read_fits.o getoifits.o  -L$(L_PATH_X11) -L$(L_PATH_PG) -L$(L_PATH_LOC) $(LIBS_FITS)



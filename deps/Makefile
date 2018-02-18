ifeq ($(OS), Windows_NT)
	SLIB = dll
else
	UNAME := $(shell uname)
	ifeq ($(UNAME), Darwin)
		SLIB = dylib
	else
		SLIB = so
	endif
endif

OBJ = Rotations.c
CFLAGS = -std=c11 -Ofast -march=native
LIBFLAGS = -shared -fPIC -lm -lgomp

all:
	make seriallib
	make parallellib

serial:
	gcc-4.9 $(CFLAGS) $(OBJ) -o rot

parallel:
	gcc-4.9 -fopenmp $(CFLAGS) $(OBJ) -o rotpar

seriallib:
	gcc-4.9 $(CFLAGS) $(LIBFLAGS) $(OBJ) -o rot.$(SLIB)

parallellib:
	gcc-4.9 -fopenmp $(CFLAGS) $(LIBFLAGS) $(OBJ) -o rotpar.$(SLIB)

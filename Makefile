CCFLAGS=-O3 -fopenmp -Wno-unused-result
PATHS=-I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/6/include/ -L/usr/lib/x86_64-linux-gnu/
LIBS=-lgsl -lgslcblas -lm 

FlowsForTheMasses: FlowsForTheMasses.cc Makefile AU_pcu.h AU_ncint.h AU_fftgrid.h AU_cosmoparam.h AU_cosmofunc.h AU_fastpt_coord.h AU_combinatorics.h  AU_fluid.h 
	g++ FlowsForTheMasses.cc -o FlowsForTheMasses $(CCFLAGS) $(PATHS) $(LIBS) 

clean:
	$(RM) FlowsForTheMasses


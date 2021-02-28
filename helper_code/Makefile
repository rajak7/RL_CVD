2d_phase: 2d_phase.o atom_property.o
	gcc -o 2d_phase 2d_phase.o atom_property.o -lm

2d_phase.o: 2d_phase.c atom_property.h 
	gcc  -c -Wall -std=c99 2d_phase.c

atom_property.o: atom_property.c atom_property.h
	gcc  -c -Wall -std=c99 atom_property.c

clean:
	rm -f *.o 2d_phase

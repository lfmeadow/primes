
CXX=mpic++
CXXFLAGS=-g -O3 -Wno-ignored-attributes -I/opt/rocm/include \
-I/work1/lmeadows/rocPRIM-install/include \
-D__HIP_PLATFORM_AMD__=1 -Wno-unused-result
OBJECTS=args.o main.o bitvector.o primes.o
primes:  $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o primes $(OBJECTS)
clean:
	rm -f primes $(OBJECTS)

args.o: args.cc
main.o: main.cc
bitvector.o: bitvector.cc
primes.o: primes.cc

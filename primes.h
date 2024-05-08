#include "hip.h"
#include "bitvector.h"
#include "isqrt.h"
#include <vector>

class Primes
{
private:
  int device;
  hipDeviceProp_t properties;
  int nCUs;
  const uint64_t m1, m2, n, nB, k, adj;
  uint64_t primecount;
  // This is the total number of collapsed iterations.
  uint64_t total_iterations;
  // bitvector (device only)
  Bitvector B;
  // strides/cumsum(strides) (device only)
  uint64_t *counts;
  // base for bitvector index
  uint64_t *bindices;

public:
  // primes from start to end
  Primes(uint64_t start, uint64_t end);

  ~Primes();

  void dump();
  void dumpsched();

  // helpers
  static __device__ uint64_t basis (uint64_t i) { return 2 * i * (i+1); }
  static __device__ uint64_t stride(uint64_t i) { return 2*i + 1; }


  void init();

  uint64_t countprimes();

  uint64_t getprimes(std::vector<uint64_t> &primes, int maxprimes);
  uint64_t getbracket(std::vector<uint64_t> &primes);

  void schedulex();

};

#include <cstdint>

// integer (floor) sqrt of uint64_t
// derived from Wikepedia int32_t C version of a well-known algorithm
// Tested only on AMD x86_64, should work anywhere if __builtin_clzll is correct.
static inline uint64_t
sqrt64(uint64_t n)
{
  if (n == 0)
    return 0;
  // X_(n+1)
  uint64_t x = n;

  // c_n
  uint64_t c = 0;

  // d_n which starts at the highest power of four <= n
  int pos = (63 - __builtin_clzll(n));
  pos &= ~1;
  uint64_t d = 1L << pos;

  // for dₙ … d₀
  while (d != 0) {
      if (x >= c + d) {      // if X_(m+1) ≥ Y_m then a_m = 2^m
          x -= c + d;        // X_m = X_(m+1) - Y_m
          c = (c >> 1) + d;  // c_(m-1) = c_m/2 + d_m (a_m is 2^m)
      }
      else {
          c >>= 1;           // c_(m-1) = c_m/2      (aₘ is 0)
      }
      d >>= 2;               // d_(m-1) = d_m/4
  }
  return c;                  // c_(-1)
}

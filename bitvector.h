#pragma once
#include <cstdint>
#include "hip.h"

// A boolean array represented as bits. Used only on device.
// Set and clear are atomic at the bit level
class Bitvector
{
private:
  uint64_t nbits;       // array size in bits
  size_t nwords;        // total number of uint64_t
  uint64_t *data;       // always on the device
  uint64_t ntrue;       // from the last time count() was called
  int nCUs;             // for kernel launch
  Bitvector *d_ptr;        // Device calls must indirect through this pointer
  // to launch kernels
  static constexpr uint32_t wgsize = 256;
  uint32_t num_teams;

  // uint64_t containing idx
  __device__ inline static uint64_t W(uint64_t idx) { return idx >> 6; }
  // bit within uint64_t containing idx
  __device__ inline static uint64_t B(uint64_t idx) { return idx & 63; }

public:
  Bitvector() : nbits(0), nwords(0), data(0) {}
  ~Bitvector();

  // initialize the bitvector to hold n bits (and launch on a device with nCUs)
  void init(uint64_t n, int ncus);

  // get the number of bits needed to get n true values.
  uint64_t firstn(int maxtrue);
  // get the index of the first and last true bits
  void firstandlast(uint64_t *vals);


  // count the number of true elements in the vector
  uint64_t count();
  // count the number of true elements from start to end
  uint64_t count_range(uint64_t, uint64_t);

  // for debugging
  void dump();

  // return device pointer that can be used to call methods on the device
  Bitvector * copy_to_device() const;

  // size in uint64_t
  __device__ size_t size() const { return nwords; }

  // get the truth value of an element. Not atomic
  __device__ bool get(uint64_t idx) const {
    if (idx >= nbits) {
      return false;
    }
    return (data[W(idx)] & (1L << B(idx))) != 0;
  }

  // set an element to true (unused)
  __device__ void set(uint64_t idx) {
    atomicOr(&data[W(idx)], 1L << B(idx));
  }

  // clear (set an element to false)
  __device__ void clr(uint64_t idx) {
#ifdef DEBUG
    if (idx >= nbits) {
      printf("BV clr Botch: idx=%lu nbits=%lu\n", idx, nbits);
      return;
    }
#endif
    atomicAnd(&data[W(idx)], ~(1UL << B(idx)));
  }
};

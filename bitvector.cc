#include <cstdio>
#include "bitvector.h"
#include "algorithm.h"
#include <rocprim/rocprim.hpp>

// kernel to set all bits to 1
__global__ static void
init_bitvector(uint64_t *data, uint64_t nwords, uint64_t nbits)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t nthreads = blockDim.x * gridDim.x;
  const uint64_t nfullwords = nbits >> 6;

  const Bounds bnd(nwords, nthreads, idx);
  for (uint64_t i = bnd.lb; i < bnd.ub; ++i) {
    if (i == nfullwords) {
      const int remaining = nbits - nfullwords*64;
      data[i] = (~uint64_t(0)) >> (64 - remaining);
    }
    else
      data[i] = ~uint64_t(0);
  }
}

// initialize the bitvector
void 
Bitvector::init(uint64_t n, int ncus)
{
  nbits = n;
  nwords = (nbits + 63) >> 6;
  ntrue = 0;
  nCUs = ncus;
  num_teams = 6 * nCUs;
  HIP_CHECK(hipMalloc(&data, nwords * sizeof(uint64_t)));
  HIP_CHECK(hipMalloc(&d_ptr, sizeof(Bitvector)));
  init_bitvector<<<dim3(num_teams), dim3(wgsize)>>>(data, nwords, nbits);
  HIP_CHECK(hipGetLastError());
}

Bitvector *
Bitvector::copy_to_device() const
{
  HIP_CHECK(hipMemcpy(d_ptr, this, sizeof(Bitvector), hipMemcpyHostToDevice));
  return d_ptr;
}

Bitvector::~Bitvector()
{
  if (data)
    HIP_CHECK(hipFree(data));
  if (d_ptr)
    HIP_CHECK(hipFree(d_ptr));
}

// FIXME this a stupid way to do it. We will come up with cumulative
// sum by word up to some small amount of storage, from the front and
// the back, and just return those bookends
uint64_t
Bitvector::firstn(int maxtrue)
{
#if 0
  uint64_t high = nwords;
  uint64_t cnt = ntrue;
  if (cnt == 0)
    cnt = count();
  while (high > 1) {
    if (maxtrue >= cnt)
      break;
    high = high /  2;
    cnt = count_range(0, high);
  }
  return cnt;
#endif
  return 0;
}

// get the indicies of the first and last true bits
void
Bitvector::firstandlast(uint64_t *vals)
{
#if 0
  uint64_t front = 0;
  uint64_t cntf = 0;
  while (cntf == 0 && front < nwords)
    cntf = count_range(0, ++front);
  printf("cntf=%lu front=%lu\n", cntf, front);
  assert(front < nwords);
  uint64_t cntb = 0;
  uint64_t back = nwords;
  while (cntb == 0 && back > 0)
    cntb = count_range(--back, nwords);
  printf("cntb=%lu back=%lu nwords=%lu\n", cntb, back, nwords);
  assert(back >= 0);
  uint64_t flags[2];
  HIP_CHECK(hipMemcpy(flags, &data[front-1], sizeof(uint64_t),
    hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(flags+1, &data[back], sizeof(uint64_t),
    hipMemcpyDeviceToHost));
  printf("flags = 0x%lx 0x%lx\n", flags[0], flags[1]);
  vals[0] = (front-1)*64 + (63-__builtin_clzll(flags[0]));
  vals[1] = (back-1)*64 + __builtin_clzll(flags[1]);
  // send these back as-is, primes module convert the indices to
  // prime numbers.
#endif
  vals[0] = vals[1] = 0;
}

void
Bitvector::dump()
{
  uint64_t *hostcopy = new uint64_t[nwords];
  HIP_CHECK(hipMemcpy(hostcopy, data, nwords*sizeof(uint64_t),
    hipMemcpyDeviceToHost));
  printf("Bdata: ");
  for (int i = 0; i < nwords; ++i)
    printf(" 0x%lx", hostcopy[i]);
  printf("\n");
  delete [] hostcopy;
}

uint64_t
Bitvector::count_range(uint64_t start, uint64_t end)
{
  auto popcnt64 = [] __device__(uint64_t a) -> uint64_t
  {
    return __popcll(a);
  };
  auto popcit = rocprim::make_transform_iterator(&data[start], popcnt64);

  size_t nbytes;
  void *ptr = 0;
  uint64_t *cntp;
  HIP_CHECK(hipMalloc(&cntp, sizeof(uint64_t)));

  HIP_CHECK(rocprim::reduce(ptr, nbytes, popcit, cntp, 0L, end-start));
  HIP_CHECK(hipMalloc(&ptr, nbytes));
  HIP_CHECK(rocprim::reduce(ptr, nbytes, popcit, cntp, 0L, end-start));
  HIP_CHECK(hipFree(ptr));
  uint64_t cnt;
  HIP_CHECK(hipMemcpy(&cnt, cntp, sizeof(uint64_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(cntp));
  return cnt;
}

uint64_t
Bitvector::count()
{
  ntrue = count_range(0, nwords);
  return ntrue;
}

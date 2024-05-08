#include "primes.h"
#include "algorithm.h"
#include <rocprim/rocprim.hpp>
#include "args.h"
#include <limits>

Primes::Primes(uint64_t start, uint64_t end) : m1(start), m2(end), n(m2-m1+1),
    nB((n-1)/2+1), k((sqrt64(m2)-1)/2), adj((m1-1)/2), device(0)
{
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipMalloc(&counts, (k+1) * sizeof(uint64_t)));
  HIP_CHECK(hipMalloc(&bindices, (k+1) * sizeof(uint64_t)));
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  HIP_CHECK(hipGetDeviceProperties(&properties, device);
  if (argvals.verbose)
    printf("Device #0 %s, %d CUs\n", properties.name,
      properties.multiProcessorCount);
  nCUs = properties.multiProcessorCount);
  if (argvals.verbose)
    printf("m1=%lu m2=%lu nB=%lu k=%lu adj=%lu\n",
      m1, m2, nB, k, adj);
}

Primes::~Primes()
{
  HIP_CHECK(hipFree(counts));
}

void
Primes::dump()
{
  uint64_t *hostcopy = new uint64_t[k];
  HIP_CHECK(hipMemcpy(hostcopy, counts, (k+1)*sizeof(uint64_t), hipMemcpyDeviceToHost));
  printf("counts: ");
  for (int i = 0; i <= k; ++i)
    printf(" %lu", hostcopy[i]);
  printf("\n");
  delete [] hostcopy;
}

static __global__ void
compute_counts(uint64_t *cnts, uint64_t *binds,
  uint64_t k, uint64_t m1, uint64_t m2)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t nthreads = blockDim.x * gridDim.x;
  const Bounds bnd(k, nthreads, idx);
  uint64_t adj = (m1-1)/2;
  auto scalem1 = [=](uint64_t b, uint64_t s) {
    uint64_t v = (m1-1) >> 1;
    if (v < b) return 0UL;
    v -= b;
    uint64_t q = v / s;
    uint64_t r = v - q * s;
    return q + (r != 0);
  };

  if (bnd.lb == 0)
    cnts[0] = 0;
  for (uint64_t i = bnd.lb; i < bnd.ub; ++i) {
    // get the number of iterations 
    uint64_t b = Primes::basis(i+1);
    uint64_t s = Primes::stride(i+1);
    uint64_t m2idx = ((m2-1)/2 - b) / s;
    uint64_t m1idx = scalem1(b, s);
    cnts[i+1] = m2idx - m1idx + 1;
    // base bidx
    binds[i+1] = b + m1idx * s - adj;
  }
}

void
Primes::init()
{
  const uint32_t wgsize = 256;
  const uint32_t num_teams = 6 * nCUs;
  compute_counts<<<dim3(num_teams), dim3(wgsize), 0, hipStreamDefault>>>(
    counts, bindices, k, m1, m2);
  HIP_CHECK(hipGetLastError());
  //dump();
  // now turn the array into a cumulative sum
  // use the rocPrim implementation
  size_t nbytes;
  void *ptr = 0;
  HIP_CHECK(rocprim::inclusive_scan(ptr, nbytes,
    counts+1, counts+1, k, rocprim::plus<uint64_t>()));
  HIP_CHECK(hipMalloc(&ptr, nbytes));
  HIP_CHECK(rocprim::inclusive_scan(ptr, nbytes,
    counts+1, counts+1, k, rocprim::plus<uint64_t>()));
  HIP_CHECK(hipFree(ptr));
  //dump();
  /* last element is total iteration count */
  HIP_CHECK(hipMemcpy(&total_iterations, &counts[k],
    sizeof(uint64_t), hipMemcpyDeviceToHost));
  //dump();

  if (argvals.verbose)
    std::cout << "number of iterations is " << total_iterations << '\n';
}

// recover loop indices from flat index
struct Bndinfo {
  const uint64_t *lb;
  uint64_t offs;
  uint64_t cnt;
  uint64_t i;
  __device__
  Bndinfo(uint64_t j, uint64_t k, const uint64_t *counts)
  {
    lb = upper_bound(&counts[1], &counts[k], j);
    i = lb - &counts[0];
    offs = j - counts[i-1];
    cnt = counts[i] - counts[i-1];
  }
  __device__ uint64_t
  clearbit(uint64_t *bindices) const
  {
    return bindices[i] + offs * Primes::stride(i);
  }
};

static __global__ void
mark_primes(
  uint64_t *counts,
  uint64_t *bindices,
  uint64_t k,
  uint64_t total_iterations,
  uint64_t m1, Bitvector *Bp)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t nthreads = blockDim.x * gridDim.x;
  const Bounds bnd(total_iterations, nthreads, idx);
  if (bnd.lb == bnd.ub)
    return;
  const uint64_t adj = (m1 - 1) / 2;
  const uint64_t nbits = Bp->size() * 64;

  // we reconstruct the original doubly-nested loops so we
  // can simplify recovering that info from the flat index

  const Bndinfo lb(bnd.lb, k, counts);
  uint64_t bidx = lb.clearbit(bindices);
  uint64_t j = bnd.lb;
  uint64_t st = Primes::stride(lb.i);
#ifdef DEBUG
  printf("j=%lu idx=%u i=%lu cnt=%lu bidx=%lu\n",
        j, idx, lb.i, lb.cnt, bidx);
#endif

  // finish this i
  uint64_t range = bnd.ub - bnd.lb;
  const uint64_t icnt = lb.cnt - lb.offs <= range ? lb.cnt - lb.offs : range;
  for (uint64_t offs = lb.offs; offs < lb.offs + icnt; ++offs) {
#ifdef DEBUG
    if (bidx  >= nbits) {
      printf(
  "j=%lu idx=%u i=%lu offs=%lu cnt=%lu bidx=%lu nbits=%lu range=%lu BOTCH\n",
          idx, lb.i, offs, lb.cnt, bidx, nbits, range);
        return;
    }
#endif
    Bp->clr(bidx);
    bidx += st;
  }
  range -= icnt;

  // do the remaining i's
  for (uint64_t i = lb.i+1; range; ++i) {
    const uint64_t st = Primes::stride(i);
    const uint64_t cnt = counts[i] - counts[i-1];
    uint64_t bidx = bindices[i];
#ifdef DEBUG
    printf("range=%lu idx=%u i=%lu cnt=%lu bidx=%lu\n",
        range, idx, i, cnt, bidx);
#endif
    const uint64_t icnt = cnt <= range ? cnt : range;
    range -= icnt;
    for (uint64_t offs = 0;  offs < icnt; ++offs) {
#ifdef DEBUG
        if (bidx >= nbits) {
          printf(
    "idx=%u i=%lu offs=%lu cnt=%lu bidx=%lu nbits=%lu range=%lu BOTCH\n",
            idx, i, offs, cnt, bidx, nbits, bnd.ub-bnd.lb);
          return;
        }
#endif
      Bp->clr(bidx);
      bidx += st;
    }
  }
}

static __global__ void
mark_primes_nojump(
  uint64_t *counts,
  uint64_t *bindices,
  uint64_t k,
  uint64_t total_iterations,
  uint64_t m1, Bitvector *Bp)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_iterations)
    return;

  Bndinfo thisone(idx, k, counts);
  auto bidx = thisone.clearbit(bindices);
  Bp->clr(bidx);
}

uint64_t
Primes::countprimes()
{
  // make the bitset
  B.init(nB, nCUs);

  const int log2wg = 8;
  const int wgsize = (1 << log2wg);
  const int wgmask = wgsize - 1;

  // if total_iterations <= ~uint32_t(0), then
  // we can launch a "no-jump" kernel.
  // kernel will need to check that it's in bounds
#if 0
  if (total_iterations <= std::numeric_limits<uint32_t>::max()) {
    const uint32_t num_teams = (total_iterations >> log2wg) +
      ((total_iterations & wgmask) != 0);
    if (argvals.verbose)
      printf("mark_primes_nojump: num_teams=%u wgsize=%u total_iterations=%lu\n",
        num_teams, wgsize, total_iterations);
    mark_primes_nojump<<<dim3(num_teams), dim3(wgsize), 0, hipStreamDefault>>>(
      counts, k, total_iterations, m1, B.copy_to_device());
  }
  else
#endif
  {
    // kernel has to do some looping. This choice of num_teams and wgsize
    // seems to be the best
    const uint64_t num_teams = 8 * nCUs;
    mark_primes<<<dim3(num_teams), dim3(wgsize), 0, hipStreamDefault>>>(
      counts, bindices, k, total_iterations, m1, B.copy_to_device());
  }
  HIP_CHECK(hipGetLastError());
  primecount = B.count();
  return primecount;
}

class Primeit
{
  uint64_t m_adj;
  const Bitvector *m_Bv;
  Primeit *d_this;

  static __device__ Primeit *prime_global;

  using ci = rocprim::counting_iterator<uint64_t, uint64_t>;

public:
  Primeit(uint64_t adj, const Bitvector *Bv):
    m_adj(adj), m_Bv(Bv->copy_to_device())
  {
    HIP_CHECK(hipMalloc(&d_this, sizeof(Primeit)));
    HIP_CHECK(hipMemcpy(d_this, this, sizeof(Primeit),
                        hipMemcpyHostToDevice));

    void *addr;
    HIP_CHECK(hipGetSymbolAddress(&addr, HIP_SYMBOL(prime_global)));
    HIP_CHECK(hipMemcpy(addr, &d_this, sizeof(void *),
      hipMemcpyHostToDevice));
  }
  ~Primeit() { HIP_CHECK(hipFree(d_this)); }
  auto input() {
    return rocprim::make_transform_iterator(
      ci(),
      []__device__(uint64_t i) -> uint64_t {
        return 2*(i + prime_global->m_adj) + 1;
      });
  }
  auto flags() {
    return rocprim::make_transform_iterator(
        ci(),
        []__device__ (uint64_t i) -> bool {
          return prime_global->m_Bv->get(i);
        });
  }
};

uint64_t
Primes::getbracket(std::vector<uint64_t> &primes)
{
   uint64_t firstlast[2];
   B.firstandlast(firstlast);
   primes.assign(&firstlast[0], &firstlast[2]);
   return 2;
}

uint64_t
Primes::getprimes(std::vector<uint64_t> &primes, int maxprimes)
{
  uint64_t *dev_primes;
  uint64_t *cntp;
  uint64_t adj_nprimes = primecount;
  uint64_t nBsearch;
  HIP_CHECK(hipMalloc(&cntp, sizeof(uint64_t)));
  if (maxprimes) {
    nBsearch = B.firstn(maxprimes);
    adj_nprimes = nBsearch;
  }
  else
    nBsearch = nB;
  if (argvals.verbose)
    printf("Searching %lu bits for %lu primes\n", nBsearch, adj_nprimes);
  HIP_CHECK(hipMalloc(&dev_primes, adj_nprimes*sizeof(uint64_t)));

  Primeit pit(adj, &B);
  void *ptr = 0;
  size_t nbytes;
  HIP_CHECK(rocprim::select(ptr, nbytes,
    pit.input(), pit.flags(),
    dev_primes, cntp,
    nBsearch));
  HIP_CHECK(hipMalloc(&ptr, nbytes));
  HIP_CHECK(rocprim::select(ptr, nbytes,
    pit.input(), pit.flags(),
    dev_primes, cntp,
    nBsearch));
  HIP_CHECK(hipFree(ptr));

  uint64_t ndevprimes;
  HIP_CHECK(hipMemcpy(&ndevprimes, cntp, sizeof(uint64_t),
    hipMemcpyDeviceToHost));
  if (argvals.verbose)
    printf("ndevprimes=%lu\n", ndevprimes);
  primes.resize(ndevprimes, 0);
  HIP_CHECK(hipMemcpy(primes.data(), dev_primes, 
    sizeof(uint64_t) * ndevprimes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(cntp));
  HIP_CHECK(hipFree(dev_primes));
  return ndevprimes;
}

__device__ Primeit *Primeit::prime_global;

#include "primes.h"
#include "args.h"
#include <iostream>
#include <vector>
#include <mpi.h>

int
main(int argc, char **argv)
{
  timers.start("parseargs");
  parse_args(argc, argv);
  uint64_t m1 = argvals.low;
  uint64_t m2 = argvals.high;
  assert(m1 > 2);
  assert(m2 > m1);
  timers.stop("parseargs");

  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  // if there's a maxrange then that is our blocksize, otherwise blocksize
  // range / nranks
  uint64_t blocksize = argvals.maxrange;
  const uint64_t range = m2 - m1 + 1;
  uint64_t nblocks;
  if (blocksize == 0) {
    // FIXME possible overflow
    assert(range < ~uint64_t(0) - nranks);
    blocksize = (range + nranks - 1) / nranks;
    nblocks = nranks;
  }
  else {
    // FIXME possible overflow
    assert(range < ~uint64_t(0) - blocksize);
    nblocks = (range + blocksize - 1) / blocksize;
  }

  uint64_t allprimes = 0;
  for (uint64_t block = rank; block < nblocks; block += nranks) {
    // this needs to be fixed for overflow
    uint64_t mystart = m1 + blocksize * block;
    uint64_t myend = mystart + blocksize - 1;
    if (myend > m2) myend = m2;

    timers.start("initprimes");
    Primes p(mystart, myend);
    p.init();
    timers.stop("initprimes");

    timers.start("countprimes");
    uint64_t nprimes = p.countprimes();
    if (argvals.verbose)
      printf("%d: blocksize %lu nprimes %lu\n", rank, blocksize, nprimes);
    uint64_t bout_total = 0;
    timers.stop("countprimes");

    timers.start("reduce");
    MPI_Reduce(&nprimes, &bout_total, 1,
      MPI_UNSIGNED_LONG, MPI_SUM, 0l, MPI_COMM_WORLD);
    allprimes += bout_total;
    timers.stop("reduce");

    if (argvals.type) {
      std::vector<uint64_t> primes;
      timers.start("getprimes");
      if (argvals.type == args::all) {
        uint64_t ndevprimes = p.getprimes(primes, 0);
      }
      else if (argvals.type == args::nprimes) {
        if (argvals.maxprimes < nprimes)
          nprimes = argvals.maxprimes;
        uint64_t ndevprimes = p.getprimes(primes, nprimes);
        if (ndevprimes > nprimes) {
          fprintf(stderr,
        "primes count mismatch with -t nprimes: %lu returned, %lu requested\n",
            ndevprimes, nprimes);
          exit(1);
        }
      }
      else {
        assert(argvals.type == args::bracket);
        uint64_t ndevprimes = p.getbracket(primes);
        assert(ndevprimes == 2);
      }
      timers.stop("getprimes");
      std::cout << rank << ": ";
      for (auto v:primes)
        std::cout << " " << v ;
      std::cout << '\n';
    }
    if (argvals.timers && rank == 0)
      timers.print();
  }
  int rem = nblocks % nranks;
  uint64_t dummy = 0;
  if (rank >= nranks -rem)
    // these ranks did not participate in the final bout in the loop
    MPI_Reduce(&dummy, &dummy, 1, MPI_UNSIGNED_LONG, MPI_SUM,
      0, MPI_COMM_WORLD);
  if (rank == 0)
    printf("There are %lu primes between %lu and %lu\n", allprimes,
      argvals.low, argvals.high);
  MPI_Finalize();
}

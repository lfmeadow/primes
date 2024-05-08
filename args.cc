#include "args.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>

args argvals;
Timers timers;

static void
usage(const char *cmd)
{
  fprintf(stderr, "Usage: %s [--verbose] [--timers] [-t|--type ptype] "
    "[-r|--range maxrange] start stop\n"
    "where ptype is {'count'|'bracket'|'all'|#primes} and"
    " [start,stop] is the closed interval to search\n", cmd);
}
void
parse_args(int argc, char **argv)
{
  while (true)
  {
    static struct option long_options[] = {
      {"verbose", no_argument, &argvals.verbose, 1},
      {"timers", no_argument, &argvals.timers, 1},
      {"type", required_argument, 0, 't'},
      {"range", required_argument, 0, 'r'},
      {0}
    };
    int option_index = 0;
    int c = getopt_long(argc, argv, "t:r:", long_options, &option_index);
    if (c == -1)
      break;
    switch (c)
    {
    case 0:
      break;
    case 't':
      switch (*optarg) {
      case 'c': argvals.type = args::cnt; break;
      case 'b': argvals.type = args::bracket; break;
      case 'a': argvals.type = args::all; break;
      default:
        argvals.type = args::nprimes;
        argvals.maxprimes = strtoul(optarg, 0, 0);
        break;
      }
    case 'r':
      argvals.maxrange = strtoul(optarg, 0, 0);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  if (optind < argc) {
    uint64_t first = strtoul(argv[optind++], 0, 0);
    if (optind < argc) {
      argvals.low = first;
      argvals.high = strtoul(argv[optind++], 0, 0);
    }
    else {
      argvals.low = 3;
      argvals.high = first;
    }
  }
  assert("start of range must be >= 3" && argvals.low >= 3);
  assert("start of range must be < end of range" && argvals.low < argvals.high);
}

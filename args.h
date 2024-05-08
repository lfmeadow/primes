#include <getopt.h>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iostream>


struct args {
  enum primesType {
    cnt = 0,    // count only
    bracket,    // first and last
    all,        // all the primes in the range        
    nprimes     // an int value (how many)
  };
  int verbose;
  int timers;
  // how to print primes
  enum primesType type;
  int maxprimes;
  // max range to consider
  uint64_t maxrange;
  // low and high values to enumerate
  uint64_t low, high;
};

extern args argvals;

void parse_args(int argc, char **argv);

class Timers {
public:
  Timers() {}

  void
  start(const std::string name) {
    auto timer = timers.find(name);
    if (timer == timers.end()) {
      timer = timers.insert(std::pair<std::string,Value>(name, Value())).first;
    }
    timer->second.start = clock_t::now();
  }
  void
  stop(const std::string name) {
    auto now = clock_t::now();
    auto timer = timers.find(name);
    if (timer == timers.end()) {
      fprintf(stderr, "stopping nonexistent timer: %s\n", name.c_str());
      exit(1);
    }
    timer->second.duration += now - timer->second.start;
  }
  void
  print()
  {
    using namespace std::literals;
    for (auto timer : timers) {
      auto milliseconds = 
        std::chrono::duration_cast<std::chrono::milliseconds>(
          timer.second.duration);
      double d = milliseconds.count() / 1e3;
      std::cout << timer.first << ":\t" << d << " s\n";
    }
  }
  void
  reset()
  {
    for (auto timer : timers)
      timer.second = Value{};
  }

private:
  using clock_t = std::chrono::steady_clock;
  using timepoint_t = std::chrono::time_point<clock_t>;
  using duration_t = std::chrono::duration<clock_t::rep, clock_t::period>;
  struct Value {
    timepoint_t start;
    duration_t duration;
  };
  std::map<std::string, Value> timers;
};

extern Timers timers;

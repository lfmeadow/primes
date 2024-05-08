// I can't seem to use std::lower_bound and std::upper_bound on the device without
// lots of whining, so here are dumb versions that work for this purpose
template <class ForwardIterator, class T>
__device__ ForwardIterator
lower_bound(ForwardIterator first, ForwardIterator last, const T& val)
{
  ForwardIterator it;
  typename std::iterator_traits<ForwardIterator>::difference_type count, step;
  count = last-first;
  while (count>0)
  {
    it = first; step=count/2; it += step;
    if (*it<val) {
      first=++it;
      count-=step+1;
    }
    else count=step;
  }
  return first;
}

template<class ForwardIt, class T>
__device__ ForwardIt
upper_bound(ForwardIt first, ForwardIt last, const T& value)
{
  ForwardIt it;
  typename std::iterator_traits<ForwardIt>::difference_type count, step;
  count = last - first;

  while (count > 0)
  {
    it = first; 
    step = count / 2; 
    it += step;

    if (!(value < *it))
    {
      first = ++it;
      count -= step + 1;
    } 
    else
      count = step;
  }

  return first;
}
// split a range amongst threads with best possible load balance
struct Bounds
{
  uint64_t lb, ub;
  __device__
  Bounds(uint64_t n, int nthreads, int thread) {
    uint64_t q = n / nthreads;
    uint64_t r = n - q*nthreads;
    if (thread < r) {
      lb = thread * (q+1);
      ub = lb + q + 1;
    }
    else {
      lb = thread * q + r;
      ub = lb + q;
    }
  }
};

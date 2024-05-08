#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>

constexpr int error_exit_code = -1;

// I do dislike this technique, it obscures the actual function calls.
// Probably need better abstrations, HIP API is really pure C... sigh.
#define HIP_CHECK(condition)  \
do { \
  const hipError_t error = condition; \
  if(error != hipSuccess) \
  {  \
    std::cerr << "An error encountered: \"" << hipGetErrorString(error) << \
    "\" at "  << __FILE__ << ':' << __LINE__ << std::endl; \
    std::exit(error_exit_code); \
  } \
} while (0)


#ifndef SUPPORTED_BATCH_ABI_H
#define SUPPORTED_BATCH_ABI_H
#include <stddef.h>
#include <stdint.h>
#define SUPPORTED_SUCCESS ((size_t)-1)
#define SUPPORTED_INVALID_ARGUMENT ((size_t)-2)
#ifdef __cplusplus
extern "C" {
#endif
/* ABI v1: IEEE Float64, field-major input[15][count], output[11][count].
   Caller owns disjoint buffers. Return SIZE_MAX on success, otherwise the first
   invalid/failed body index, or SUPPORTED_INVALID_ARGUMENT for bad call arguments.
   Discard every output after an update failure.
   Validate inputs once before running. No pointers retained, allocations,
   hidden sums, contact discovery or world integration. Parallelism explicit. */
uint32_t supported_abi_version(void);
uint32_t supported_input_fields(void);
uint32_t supported_output_fields(void);
size_t supported_validate(const double* input,size_t count);
size_t supported_batch(const double* input,double* output,size_t count,int threads);
int supported_max_threads(void);
#ifdef __cplusplus
}
#endif
#endif

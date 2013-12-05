// With mods by Brian

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "P2P_Compressed.hpp"
#include "fmmtl/config.hpp"


#define THRDS_PER_BLK 256
// This function has been altered to conform to the use of shared memory and as such,
// no longer offers a valid baseline comparison without using shared memory.
//#define TEST_SMEM 1

// A quick class to time gpu kernels using device events
struct StopWatch {
  cudaEvent_t startTime, stopTime;
  StopWatch()  { cudaEventCreate(&startTime); cudaEventCreate(&stopTime); }
  ~StopWatch() { cudaEventDestroy(startTime); cudaEventDestroy(stopTime); }
  inline void start() { cudaEventRecord(startTime,0); }
  inline double stop() { return elapsed(); }
  inline double elapsed() {
    cudaEventRecord(stopTime,0);
    cudaEventSynchronize(stopTime);
    float result;
    cudaEventElapsedTime(&result, startTime, stopTime);
    return result/1000.0;    // 1000 mSec per Sec
  }
};

inline int cudaInit(int device = 0) {
  StopWatch initTimer;
  initTimer.start();

  int count;
  cudaGetDeviceCount(&count);
  if (count == 0) {
    std::cerr << "Error: No devices supporting CUDA" << std::endl;
    exit(1);
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (prop.major < 1) {
    std::cerr << "Error: " << prop.name << " doesn't support CUDA." << std::endl;
    exit(1);
  }

  device = min(count-1, max(0, device));
  cudaSetDevice(device);
  std::cerr << "Initializing " << prop.name << "... ";

  int* temp;
  cudaMalloc((void**)&temp, sizeof(int));
  FMMTL_CUDA_CHECK;
  cudaFree(temp);

  std::cerr << initTimer.stop() << "s" << std::endl << std::endl;
  return 1;
}


template <typename Container>
inline typename Container::value_type* gpu_copy(const Container& c) {
	typedef typename Container::value_type c_value;
	// Allocate
	thrust::device_ptr<c_value> dptr = thrust::device_malloc<c_value>(c.size());
	// Copy
	//thrust::uninitialized_copy(c.begin(), c.end(), dptr);
	thrust::copy(c.begin(), c.end(), dptr);
	// Return
	return thrust::raw_pointer_cast(dptr);
}

template <typename T>
inline void gpu_free(T* p) {
	thrust::device_free(thrust::device_pointer_cast<void>(p));
}

template <typename Kernel,
typename RandomAccessIterator1,  // pair<uint,uint>
typename RandomAccessIterator2,  // Chained uint
typename RandomAccessIterator3>  // pair<uint,uint>
__global__ void
blocked_p2p(Kernel K,  // The Kernel to apply
		// BlockIdx -> pair<uint,uint> target range
		RandomAccessIterator1 target_range,
		// BlockIdx,BlockIdx+1 -> pair<uint,uint> into source_range
		RandomAccessIterator2 source_range_ptr,
		// Idx -> pair<uint,uint> source range
		RandomAccessIterator3 source_range,
		const typename Kernel::source_type* source,
		const typename Kernel::charge_type* charge,
		const typename Kernel::target_type* target,
		typename Kernel::result_type* result) {
	typedef Kernel kernel_type;
	typedef typename kernel_type::source_type source_type;
	typedef typename kernel_type::charge_type charge_type;
	typedef typename kernel_type::target_type target_type;
	typedef typename kernel_type::result_type result_type;

	// Get the target range this block is responsible for
	const thrust::pair<unsigned,unsigned> t_range = target_range[blockIdx.x];
	const unsigned t_first = t_range.first;
	const unsigned t_last  = t_range.second;
	const unsigned num_targets = t_last-t_first;

	// Get the range of source ranges this block is responsible for
	RandomAccessIterator3 sr_first = source_range + source_range_ptr[blockIdx.x+0];
	RandomAccessIterator3 sr_last  = source_range + source_range_ptr[blockIdx.x+1];

//#ifdef TEST_SMEM
	__shared__ source_type so_sh[THRDS_PER_BLK];
	__shared__ charge_type ch_sh[THRDS_PER_BLK];
//#endif

	// Parallel for each target until the last.
	// If at least one thread has a valid target this iteration, all threads get in. This is
  // so we ensure that all threads are available for use in populating shared memory.
	for (unsigned t_base = t_first; t_base < t_last; t_base += blockDim.x) {
		const unsigned t_current = t_base + threadIdx.x;

		target_type t;
    result_type r;
		// This guard is necessary since we've allowed all threads into the loop. Get used to seeing it...
		if (t_current < t_last) {
			t = target[t_current];
      r = result_type();
    }

		// For each source range (there must be at least one)
		do {
			// Get the range
			const thrust::pair<unsigned,unsigned> s_range = *sr_first;
			const unsigned s_first = s_range.first;
			const unsigned s_last  = s_range.second;
			const unsigned num_sources = s_last - s_first;

			// Loop over sources in this range (which could potentially outnumber threads) block by block.
			// Model this loop as above to ensure enough threads are available for all valid targets and
      // to ease future inversion of the two loops.
			for (unsigned s_base = s_first; s_base < s_last; s_base += blockDim.x) {
				const unsigned s_current = s_base + threadIdx.x;

				__syncthreads(); // Ensure the last iteration has been completed by all warps before altering the data
				// Load a chunk of source data into shared memory
				if (s_current < s_last) {
					so_sh[threadIdx.x] = source[s_current];
					ch_sh[threadIdx.x] = charge[s_current];
				}
				__syncthreads();

				// Filter out any threads that might've been useful for populating shared memory, but have no valid target to compute a result for.
				if (t_current < t_last) {
					// Calculate a partial result based on the charges and sources in shared memory
					for (unsigned k = 0; k < THRDS_PER_BLK; k++) {
						// What a shitty, unimaginative variable name... Oh, well!
            // TODO: Opportunity to decrease potential bank conflicts here using modulo offsetted indexing
						const unsigned s_current2 = s_base + k;

						// If we're not out of bounds, do the calculation
						if (s_current2 < s_last) {
							const source_type s = so_sh[k];
							const charge_type c = ch_sh[k];

							r += K(t,s) * c;
						}
					}
				}
			} // Loop over sources by block

			++sr_first;
		} while(sr_first < sr_last);

		// Assign the result
		if (t_current < t_last)
			result[t_current] += r;
	}
}

//			if (0 && threadIdx.x < 5) {
//				printf("block %d thrd %d dim %d s_first %d s_last %d del %d\n",
//						blockIdx.x, threadIdx.x, blockDim.x, s_first, s_last, s_last-s_first);
//			}
//
//
//#ifdef TEST_SMEM
//			if (num_sources > THRDS_PER_BLK) {
//				printf("ERROR! Threads/block! %d\n", num_sources);
//				return;
//			}
//
//			// Load the current source range into shared memory
//			use_smem = true;
//			if (threadIdx.x < num_sources) {
//				so_sh[threadIdx.x] = source[s_first+threadIdx.x];
//				ch_sh[threadIdx.x] = charge[s_first+threadIdx.x];
//			}
//			__syncthreads();
//
//			// fill smem with sources and charges. each thrd handles a source.
//			if ((threadIdx.x < num_sources)
//					&& (num_sources <= THRDS_PER_BLK )) {
//				use_smem = true;
//
//				so_sh[ threadIdx.x ] = source[ s_first+threadIdx.x ];
//				ch_sh[ threadIdx.x ] = charge[ s_first+threadIdx.x ];
//
//				if(threadIdx.x == 3 && blockIdx.x == 2) {
//					printf("cache: block %d thrd %d dim %d s_first %d s_last %d del %d val\n",
//							blockIdx.x, threadIdx.x, blockDim.x, s_first, s_last,	num_sources);
//					print_vec(K, so_sh[threadIdx.x]);
//					print_vec(K, source[s_first + threadIdx.x]);
//				}
//			}
//			__syncthreads();
//#endif
//
//			// Filter out any threads that might've been useful for populating shared memory, but have valid target to compute a result for.
//			if (t_current < t_last) {
//				// For each source in the source range
//				for (unsigned s_current = s_first; s_current < s_last; ++s_current) {
//#ifndef TEST_SMEM
//					const source_type s = source[s_current];
//					const charge_type c = charge[s_current];
//#else
//					if (use_smem) {
//						const source_type s = so_sh[s_current - s_first];
//						const charge_type c = ch_sh[s_current - s_first];
//
//						if(threadIdx.x == 3 && blockIdx.x == 2) {
//							printf("use: block %d thrd %d dim %d s_current %d s_last %d idx %d val\n",
//									blockIdx.x, threadIdx.x, blockDim.x, s_current, s_last,
//									s_current - s_first);
//							print_vec(K, s);
//							print_vec(K, source[s_current]);
//						}
//
//						if (s != source[s_current])
//							printf("ERROR: %d %d\n", blockIdx.x, threadIdx.x);
//
//						r += K(t,s) * c;
//					} else { // use global mem
//						const source_type s = source[s_current];
//						const charge_type c = charge[s_current];
//						r += K(t,s) * c;
//					}
//#endif
//#ifndef TEST_SMEM
//					r += K(t,s) * c;
//#endif
//				} // for each source
//			}
//			++sr_first;
//
//#ifdef TEST_SMEM
//			use_smem = false;
//			__syncthreads();
//#endif
//		} while(sr_first < sr_last);
//
//		// Assign the result
//		if (t_current < t_last)
//			result[t_current] += r;
//	}
//}


struct Data {
	unsigned num_sources;
	unsigned num_targets;
	unsigned num_threads_per_block;
	unsigned num_blocks;
	Data(unsigned s, unsigned t, unsigned b)
	: num_sources(s),
	  num_targets(t),
	  //        num_threads_per_block(256),
	  num_threads_per_block( THRDS_PER_BLK ),
	  num_blocks(b) {
	}
};

template <typename Kernel>
P2P_Compressed<Kernel>::P2P_Compressed()
: data_(0) {
  cudaInit();
}

template <typename Kernel>
P2P_Compressed<Kernel>::P2P_Compressed(
		std::vector<std::pair<unsigned,unsigned> >& target_ranges,
		std::vector<unsigned>& source_range_ptrs,
		std::vector<std::pair<unsigned,unsigned> >& source_ranges,
		const std::vector<typename Kernel::source_type>& sources,
		const std::vector<typename Kernel::target_type>& targets)
		: data_(new Data(sources.size(), targets.size(), target_ranges.size())),
		  target_ranges_(gpu_copy(target_ranges)),
		  source_range_ptrs_(gpu_copy(source_range_ptrs)),
		  source_ranges_(gpu_copy(source_ranges)),
		  sources_(gpu_copy(sources)),
		  targets_(gpu_copy(targets)) {
  cudaInit();
}

template <typename Kernel>
P2P_Compressed<Kernel>::~P2P_Compressed() {
	delete data_;
	gpu_free(target_ranges_);
	gpu_free(source_range_ptrs_);
	gpu_free(source_ranges_);
	gpu_free(sources_);
	gpu_free(targets_);
}

template <typename Kernel>
void P2P_Compressed<Kernel>::execute(
		const Kernel& K,
		const std::vector<typename Kernel::charge_type>& charges,
		std::vector<typename Kernel::result_type>& results) {
	typedef Kernel kernel_type;
	typedef typename kernel_type::source_type source_type;
	typedef typename kernel_type::target_type target_type;
	typedef typename kernel_type::charge_type charge_type;
	typedef typename kernel_type::result_type result_type;

	// XXX: Using a device_vector here was giving "floating point exceptions"...
	// XXX: device_vector doesn't like the Vec?
	charge_type* d_charges = gpu_copy(charges);
	result_type* d_results = gpu_copy(results);

	Data* data = reinterpret_cast<Data*>(data_);
	const unsigned num_tpb    = data->num_threads_per_block;
	const unsigned num_blocks = data->num_blocks;

#if defined(FMMTL_DEBUG)
	std::cout << "Launching GPU Kernel: (blocks, threads/block) = ("
			<< num_blocks << ", " << num_tpb << ")" << std::endl;
#endif

	// Launch kernel <<<grid_size, block_size>>>
	blocked_p2p<<<num_blocks,num_tpb>>>(
			K,
			target_ranges_,
			source_range_ptrs_,
			source_ranges_,
			sources_,
			//thrust::raw_pointer_cast(d_charges.data()),
			d_charges,
			targets_,
			d_results);
	//thrust::raw_pointer_cast(d_results.data()));
	FMMTL_CUDA_CHECK;

	// Copy results back
	thrust::device_ptr<result_type> d_results_ptr = thrust::device_pointer_cast(d_results);
	thrust::copy(d_results_ptr, d_results_ptr + results.size(), results.begin());

	gpu_free(d_results);
	gpu_free(d_charges);
}


/** A functor that maps blockidx -> (target_begin,target_end) */
template <unsigned BLOCKSIZE>
class block_range
		: public thrust::unary_function<unsigned,
		  thrust::pair<unsigned,unsigned> > {
	unsigned num_targets_;
		  public:
	__host__ __device__
	block_range(unsigned num_targets) : num_targets_(num_targets) {}
	__device__
	thrust::pair<unsigned,unsigned> operator()(unsigned blockidx) const {
		unsigned start_block = blockidx * BLOCKSIZE;
		return thrust::make_pair(start_block,
				min(start_block + BLOCKSIZE, num_targets_));
	}
};

template <typename Kernel>
void
P2P_Compressed<Kernel>::execute(const Kernel& K,
		const std::vector<source_type>& s,
		const std::vector<charge_type>& c,
		const std::vector<target_type>& t,
		std::vector<result_type>& r) {
  cudaInit();
	typedef Kernel kernel_type;
	typedef typename kernel_type::source_type source_type;
	typedef typename kernel_type::target_type target_type;
	typedef typename kernel_type::charge_type charge_type;
	typedef typename kernel_type::result_type result_type;

	source_type* d_sources = gpu_copy(s);
	charge_type* d_charges = gpu_copy(c);
	target_type* d_targets = gpu_copy(t);
	result_type* d_results = gpu_copy(r);

	// XXX: device_vector doesn't like our vector?
	//thrust::device_vector<source_type> d_sources(s);
	//thrust::device_vector<charge_type> d_charges(c);
	//thrust::device_vector<target_type> d_targets(t);
	//thrust::device_vector<result_type> d_results(r);

	//const unsigned num_tpb    = 256;
	const unsigned num_tpb    = THRDS_PER_BLK;
	const unsigned num_blocks = (t.size() + num_tpb - 1) / num_tpb;

#if defined(FMMTL_DEBUG)
	std::cout << "Launching GPU Kernel: (blocks, threads/block) = ("
			<< num_blocks << ", " << num_tpb << ")" << std::endl;
#endif

	// Launch kernel <<<grid_size, block_size>>>
	blocked_p2p<<<num_blocks, num_tpb>>>(
			K,
			thrust::make_transform_iterator(thrust::make_counting_iterator(0),
					block_range<num_tpb>(t.size())),
					thrust::make_constant_iterator(0),
					thrust::make_constant_iterator(thrust::make_pair(0,s.size())),
					d_sources,
					d_charges,
					d_targets,
					d_results);
	//thrust::raw_pointer_cast(d_sources.data()),
	//thrust::raw_pointer_cast(d_charges.data()),
	//thrust::raw_pointer_cast(d_targets.data()),
	//thrust::raw_pointer_cast(d_results.data()));
	FMMTL_CUDA_CHECK;

	// Copy results back and assign
	thrust::device_ptr<result_type> d_results_ptr = thrust::device_pointer_cast(d_results);
	thrust::copy(d_results_ptr, d_results_ptr + r.size(), r.begin());

	gpu_free(d_sources);
	gpu_free(d_charges);
	gpu_free(d_targets);
	gpu_free(d_results);
}

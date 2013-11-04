#pragma once
/** @file EvalP2P_GPU.hpp
 * @brief Header file for the GPU_P2P class.
 *
 * Note: This header file is compiled with nvcc and must use C++03.
 */

#include <vector>

#include <iostream>

template <typename Kernel>
class P2P_Compressed {
 public:
  typedef Kernel kernel_type;

  // No KernelTraits here... only C++03
  typedef typename kernel_type::source_type source_type;
  typedef typename kernel_type::target_type target_type;
  typedef typename kernel_type::charge_type charge_type;
  typedef typename kernel_type::result_type result_type;

  // Supporting data
  void* data_;

  // Device data for P2P computation
  std::pair<unsigned,unsigned>* target_ranges_;
  unsigned* source_range_ptrs_;
  std::pair<unsigned,unsigned>* source_ranges_;

  // Device source and target arrays
  source_type* sources_;
  target_type* targets_;

  P2P_Compressed();

	P2P_Compressed(std::vector<std::pair<unsigned,unsigned> >& target_ranges,
                 std::vector<unsigned>& target_ptrs,
                 std::vector<std::pair<unsigned,unsigned> >& source_ranges,
                 const std::vector<source_type>& sources,
                 const std::vector<target_type>& targets);

  ~P2P_Compressed();

  void execute(const Kernel& K,
               const std::vector<charge_type>& charges,
               std::vector<result_type>& results);

  static void execute(const Kernel& K,
                      const std::vector<source_type>& s,
                      const std::vector<charge_type>& c,
                      const std::vector<target_type>& t,
                      std::vector<result_type>& r);

  /** Construct a P2P_Compressed object by taking
   * associated source ranges and target ranges and constructing a compressed
   * representation.
   *
   * @param srfirst,srlast  A range of source ranges
   * @param trfirst         A range of target ranges
   *                          The source/target ranges are associated
   * @param sources         The sources that the source ranges map into
   * @param targets         The targets that the target ranges map into
   *
   * @pre Target ranges are disjoint. No two target ranges overlap.
   *
   * @note Creates a CSR-like compressed representation of the blocked matrix
   *
   * TODO: Clean up...
   */
  template <class SourceRangeIter, class TargetRangeIter>
  static
  P2P_Compressed<Kernel>*
  make(SourceRangeIter srfirst, SourceRangeIter srlast,
       TargetRangeIter trfirst,
       const std::vector<typename Kernel::source_type>& sources,
       const std::vector<typename Kernel::target_type>& targets) {
    unsigned num_targets = targets.size();
    //unsigned num_sources = sources.size();
    unsigned num_box_pairs = srlast - srfirst;

    // Interaction list for each target box
    // (target_first,target_last) -> {(source_first, source_last), ...}
    // TODO: faster?
    typedef std::pair<unsigned, unsigned> upair;
    std::vector<std::vector<upair> > target2sources(num_targets);
    // A list of target ranges we've seen: {(target_first, target_last), ...}
    std::vector<upair> target_ranges;

    for ( ; srfirst != srlast; ++srfirst, ++trfirst) {
      upair s_range = *srfirst;
      upair t_range = *trfirst;

      unsigned i_begin = t_range.first;
      unsigned i_end   = t_range.second;

      unsigned j_begin = s_range.first;
      unsigned j_end   = s_range.second;

      // If this is the first time we've seen this target range, record it
      if (target2sources[i_begin].empty())
        target_ranges.push_back(upair(i_begin, i_end));

      target2sources[i_begin].push_back(upair(j_begin,j_end));
    }

    unsigned num_target_ranges = target_ranges.size();

    // Construct a compressed interaction list
    std::vector<unsigned> target_ptr(num_target_ranges + 1);
    target_ptr[0] = 0;
    std::vector<upair> source_ranges(num_box_pairs);
    std::vector<upair>::iterator source_ranges_curr = source_ranges.begin();

    // For all the target ranges
    for (unsigned k = 0; k < num_target_ranges; ++k) {
      // Copy the source ranges that interact with the kth target range
      unsigned i_begin = target_ranges[k].first;
      source_ranges_curr = std::copy(target2sources[i_begin].begin(),
                                     target2sources[i_begin].end(),
                                     source_ranges_curr);

      // Record the stop index
      target_ptr[k+1] = source_ranges_curr - source_ranges.begin();
    }

    // Sanity checking
    FMMTL_ASSERT(target_ptr.back() == source_ranges.size());
    FMMTL_ASSERT(source_ranges_curr == source_ranges.end());

    return new P2P_Compressed<Kernel>(target_ranges,
                                      target_ptr,
                                      source_ranges,
                                      sources,
                                      targets);
  }
};

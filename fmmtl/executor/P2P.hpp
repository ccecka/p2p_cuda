#pragma once
/** @file P2P.hpp
 * @brief Dispatch methods for the P2P stage
 *
 */

#include <iterator>
#include <type_traits>

struct P2P
{
	/** Asymmetric block P2P using the evaluation operator
	 * r_i += sum_j K(t_i, s_j) * c_j
	 *
	 * @param[in] ...
	 */
	template <typename Kernel,
	          typename SourceIter, typename ChargeIter,
	          typename TargetIter, typename ResultIter>
	inline static void
  block_eval(const Kernel& K,
             SourceIter s_first, SourceIter s_last, ChargeIter c_first,
             TargetIter t_first, TargetIter t_last, ResultIter r_first)
  {
    typedef typename Kernel::source_type source_type;
    typedef typename Kernel::charge_type charge_type;
    typedef typename Kernel::target_type target_type;
    typedef typename Kernel::result_type result_type;

    // Sanity checks to make sure the types make sense
    FMMTL_STATIC_ASSERT((std::is_same<source_type,
                         typename std::iterator_traits<SourceIter>::value_type
                         >::value), "SourceIter::value_type != source_type");
    FMMTL_STATIC_ASSERT((std::is_same<charge_type,
                         typename std::iterator_traits<ChargeIter>::value_type
                         >::value), "ChargeIter::value_type != charge_type");
    FMMTL_STATIC_ASSERT((std::is_same<target_type,
                         typename std::iterator_traits<TargetIter>::value_type
                         >::value), "TargetIter::value_type != target_type");
    FMMTL_STATIC_ASSERT((std::is_same<result_type,
                         typename std::iterator_traits<ResultIter>::value_type
                         >::value), "ResultIter::value_type != result_type");

    for ( ; t_first != t_last; ++t_first, ++r_first) {
      const target_type& t = *t_first;
      result_type& r       = *r_first;

      SourceIter si = s_first;
      ChargeIter ci = c_first;
      for ( ; si != s_last; ++si, ++ci)
        r += K(t,*si) * (*ci);
    }
  }
};

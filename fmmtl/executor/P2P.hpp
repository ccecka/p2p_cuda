#pragma once
/** @file P2P.hpp
 * @brief Dispatch methods for the P2P stage
 *
 */

#include "fmmtl/KernelTraits.hpp"
#include <iterator>
#include <type_traits>

struct P2P
{
  /** Dual-Evaluation dispatch
   */
  template <typename Kernel>
  inline static
  typename std::enable_if<KernelTraits<Kernel>::has_eval_op &
                          !KernelTraits<Kernel>::has_transpose>::type
  symm_eval(const Kernel& K,
            const typename Kernel::source_type& p1,
            const typename Kernel::charge_type& c1,
            typename Kernel::result_type& r1,
            const typename Kernel::source_type& p2,
            const typename Kernel::charge_type& c2,
            typename Kernel::result_type& r2)
  {
    r1 += K(p1,p2) * c2;
    r2 += K(p2,p1) * c1;
  }

  /** Dual-Evaluation dispatch
   */
  template <typename Kernel>
  inline static
  typename std::enable_if<KernelTraits<Kernel>::has_eval_op &
                          KernelTraits<Kernel>::has_transpose>::type
  symm_eval(const Kernel& K,
            const typename Kernel::source_type& p1,
            const typename Kernel::charge_type& c1,
            typename Kernel::result_type& r1,
            const typename Kernel::source_type& p2,
            const typename Kernel::charge_type& c2,
            typename Kernel::result_type& r2)
  {
    typedef typename Kernel::kernel_value_type kernel_value_type;

    kernel_value_type k12 = K(p1,p2);
    r1 += k12 * c2;
    kernel_value_type k21 = K.transpose(k12);
    r2 += k21 * c1;
  }

	/** Asymmetric block P2P dispatch
	 */
	template <typename Kernel,
	          typename SourceIter, typename ChargeIter,
	          typename TargetIter, typename ResultIter>
	inline static
	typename std::enable_if<KernelTraits<Kernel>::has_vector_P2P_asymm>::type
	block_eval(const Kernel& K,
             SourceIter s_first, SourceIter s_last, ChargeIter c_first,
             TargetIter t_first, TargetIter t_last, ResultIter r_first)
	{
		K.P2P(s_first, s_last, c_first,
		      t_first, t_last, r_first);
	}

	/** Asymmetric block P2P using the evaluation operator
	 * r_i += sum_j K(t_i, s_j) * c_j
	 *
	 * @param[in] ...
	 */
	template <typename Kernel,
	          typename SourceIter, typename ChargeIter,
	          typename TargetIter, typename ResultIter>
	inline static
  typename std::enable_if<KernelTraits<Kernel>::has_eval_op &
                          !KernelTraits<Kernel>::has_vector_P2P_asymm>::type
  block_eval(const Kernel& K,
             SourceIter s_first, SourceIter s_last, ChargeIter c_first,
             TargetIter t_first, TargetIter t_last, ResultIter r_first)
  {
    typedef typename Kernel::source_type source_type;
    typedef typename Kernel::charge_type charge_type;
    typedef typename Kernel::target_type target_type;
    typedef typename Kernel::result_type result_type;

    // TODO?
    // Optimize on if(std::iterator_traits<All Iters>::iterator_category == random_access_iterator)
    // to eliminate multiple increments

    static_assert(std::is_same<source_type,
                  typename std::iterator_traits<SourceIter>::value_type
                  >::value, "SourceIter::value_type != Kernel::source_type");
    static_assert(std::is_same<charge_type,
                  typename std::iterator_traits<ChargeIter>::value_type
                  >::value, "ChargeIter::value_type != Kernel::charge_type");
    static_assert(std::is_same<target_type,
                  typename std::iterator_traits<TargetIter>::value_type
                  >::value, "TargetIter::value_type != Kernel::target_type");
    static_assert(std::is_same<result_type,
                  typename std::iterator_traits<ResultIter>::value_type
                  >::value, "ResultIter::value_type != Kernel::result_type");

    for ( ; t_first != t_last; ++t_first, ++r_first) {
      const target_type& t = *t_first;
      result_type& r       = *r_first;

      SourceIter si = s_first;
      ChargeIter ci = c_first;
      for ( ; si != s_last; ++si, ++ci)
        r += K(t,*si) * (*ci);
    }
  }

  /** Symmetric off-diagonal block P2P dispatch
   * @pre source_type == target_type
   */
  template <typename Kernel,
            typename SourceIter, typename ChargeIter, typename ResultIter>
  inline static
  typename std::enable_if<KernelTraits<Kernel>::has_vector_P2P_symm>::type
  block_eval(const Kernel& K,
             SourceIter p1_first, SourceIter p1_last, ChargeIter c1_first,
             ResultIter r1_first,
             SourceIter p2_first, SourceIter p2_last, ChargeIter c2_first,
             ResultIter r2_first)
  {
    K.P2P(p1_first, p1_last, c1_first,
          p2_first, p2_last, c2_first,
          r1_first, r2_first);
  }

  /** Symmetric off-diagonal block P2P using the evaluation operator
   * r2_i += sum_j K(p2_i, p1_j) * c1_j
   * r1_j += sum_i K(p1_j, p2_i) * c2_i
   *
   * @param[in] ...
   * @pre source_type == target_type
   * @pre For all i,j we have p1_i != p2_j
   */
  template <typename Kernel,
            typename SourceIter, typename ChargeIter, typename ResultIter>
  inline static
  typename std::enable_if<!KernelTraits<Kernel>::has_vector_P2P_symm>::type
  block_eval(const Kernel& K,
             SourceIter p1_first, SourceIter p1_last, ChargeIter c1_first,
             ResultIter r1_first,
             SourceIter p2_first, SourceIter p2_last, ChargeIter c2_first,
             ResultIter r2_first)
  {
    typedef typename Kernel::source_type source_type;
    typedef typename Kernel::charge_type charge_type;
    typedef typename Kernel::target_type target_type;
    typedef typename Kernel::result_type result_type;

    static_assert(std::is_same<source_type,
                  typename std::iterator_traits<SourceIter>::value_type
                  >::value, "SourceIter::value_type != Kernel::source_type");
    static_assert(std::is_same<charge_type,
                  typename std::iterator_traits<ChargeIter>::value_type
                  >::value, "ChargeIter::value_type != Kernel::charge_type");
    static_assert(std::is_same<target_type,
                  typename std::iterator_traits<SourceIter>::value_type
                  >::value, "SourceIter::value_type != Kernel::target_type");
    static_assert(std::is_same<result_type,
                  typename std::iterator_traits<ResultIter>::value_type
                  >::value, "ResultIter::value_type != Kernel::result_type");

    // TODO
    // Optimize on random_access_iterator?

    for ( ; p1_first != p1_last; ++p1_first, ++c1_first, ++r1_first) {
      const source_type& p1 = *p1_first;
      const charge_type& c1 = *c1_first;
      result_type& r1       = *r1_first;

      SourceIter p2i = p2_first;
      ChargeIter c2i = c2_first;
      ResultIter r2i = r2_first;
      for ( ; p2i != p2_last; ++p2i, ++c2i, ++r2i)
        P2P::symm_eval(K, p1, c1, r1, *p2i, *c2i, *r2i);
    }
  }

  /** Symmetric diagonal block P2P using the evaluation operator
   * r_i += sum_j K(p_i, p_j) * c_j
   *
   * @pre source_type == target_type
   */
  template <typename Kernel,
            typename SourceIter, typename ChargeIter, typename ResultIter>
  inline static
  typename std::enable_if<!KernelTraits<Kernel>::has_vector_P2P_symm>::type
  block_eval(const Kernel& K,
             SourceIter p_first, SourceIter p_last,
             ChargeIter c_first, ResultIter r_first)
  {
    typedef typename Kernel::source_type source_type;
    typedef typename Kernel::charge_type charge_type;
    typedef typename Kernel::target_type target_type;
    typedef typename Kernel::result_type result_type;

    static_assert(std::is_same<source_type, target_type>::value,
                  "source_type != target_type in symmetric P2P");
    static_assert(std::is_same<source_type,
                               typename SourceIter::value_type>::value,
                  "SourceIter::value_type != Kernel::source_type");
    static_assert(std::is_same<charge_type,
                               typename ChargeIter::value_type>::value,
                  "ChargeIter::value_type != Kernel::charge_type");
    static_assert(std::is_same<result_type,
                               typename ResultIter::value_type>::value,
                  "ResultIter::value_type != Kernel::result_type");

    // TODO
    // Optimize on random_access_iterator?

    SourceIter pi = p_first;
    ChargeIter ci = c_first;
    ResultIter ri = r_first;
    for ( ; pi != p_last; ++pi, ++ci, ++ri) {
      const source_type& p = *pi;
      const charge_type& c = *ci;
      result_type& r       = *ri;

      // The off-diagonal elements
      SourceIter pj = p_first;
      ChargeIter cj = c_first;
      ResultIter rj = r_first;
      for ( ; pj != pi; ++pj, ++cj, ++rj)
        P2P::symm_eval(K, p, c, r, *pj, *cj, *rj);

      // The diagonal element
      r += K(p,p) * c;
    }
  }


  //////////////////////////////////////
  /////// Context Dispatchers //////////
  //////////////////////////////////////

  struct ONE_SIDED {};
  struct TWO_SIDED {};

  /** Asymmetric P2P
   */
  template <typename Context>
  inline static void eval(Context& c,
                          const typename Context::source_box_type& source,
                          const typename Context::target_box_type& target,
                          const ONE_SIDED&)
  {
#if defined(FMMTL_DEBUG)
    std::cout << "P2P:"
              << "\n  " << source
              << "\n  " << target << std::endl;
#endif

    P2P::block_eval(c.kernel(),
                    c.source_begin(source), c.source_end(source),
                    c.charge_begin(source),
                    c.target_begin(target), c.target_end(target),
                    c.result_begin(target));
  }

  /** Symmetric P2P
   */
  template <typename Context>
  inline static void eval(Context& c,
                          const typename Context::source_box_type& box1,
                          const typename Context::target_box_type& box2,
                          const TWO_SIDED&)
  {
#if defined(FMMTL_DEBUG)
    std::cout << "P2P:"
              << "\n  " << box1
              << "\n  " << box2 << std::endl;
    std::cout << "P2P:"
              << "\n  " << box2
              << "\n  " << box1 << std::endl;
#endif

    P2P::block_eval(c.kernel(),
                    c.source_begin(box1), c.source_end(box1),
                    c.charge_begin(box1), c.result_begin(box1),
                    c.target_begin(box2), c.target_end(box2),
                    c.charge_begin(box2), c.result_begin(box2));
  }

  /** Symmetric P2P
   */
  template <typename Context>
  inline static void eval(Context& c,
                          const typename Context::source_box_type& box)
  {
#if defined(FMMTL_DEBUG)
    std::cout << "P2P:"
              << "\n  " << box << std::endl;
#endif

    P2P::block_eval(c.kernel(),
                    c.source_begin(box), c.source_end(box),
                    c.charge_begin(box), c.result_begin(box),
                    c.target_begin(box), c.target_end(box),
                    c.charge_begin(box), c.result_begin(box));
  }
};



/**
 * Batched P2P methods
 **/

#include <cmath>

#include "P2P_Compressed.hpp"

#define BOOST_UBLAS_NDEBUG
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
namespace ublas = boost::numeric::ublas;

#include <unordered_map>

/** A lazy P2P evaluator which saves a list of pairs of boxes
 * That are sent to the P2P dispatcher on demand.
 */
template <typename Context>
class P2P_Batch {
  //! Kernel type
  typedef typename Context::kernel_type kernel_type;
  //! Kernel value type
  typedef typename Context::kernel_value_type kernel_value_type;

  //! Type of box
  typedef typename Context::source_box_type source_box_type;
  typedef typename Context::target_box_type target_box_type;

  //! Box list for P2P interactions    TODO: could further compress these...
  typedef std::pair<source_box_type, target_box_type> box_pair;

  typedef std::vector<box_pair> p2p_container;
  p2p_container p2p_list;

  // For now, only use for GPU...
  P2P_Compressed<kernel_type>* p2p_compressed;

 public:
  P2P_Batch() : p2p_compressed(nullptr) {}
  ~P2P_Batch() {
    delete p2p_compressed;
  }

  /** Insert a source-target box interaction to the interaction list */
  void insert(const source_box_type& s, const target_box_type& t) {
    p2p_list.push_back(std::make_pair(s,t));
  }

  /** Compute all interations in the interaction list */
  void execute(Context& c) {
#if FMMTL_NO_CUDA
    auto b_end = p2p_list.end();
    //#pragma omp parallel for//   TODO: Make thread safe!
    for (auto bi = p2p_list.begin(); bi < b_end; ++bi) {
      auto& b2b = *bi;
      P2P::eval(c, b2b.first, b2b.second, P2P::ONE_SIDED());
    }
#else
    if (p2p_compressed == nullptr)
      p2p_compressed =
          P2P_Compressed<kernel_type>::make(c, p2p_list.begin(), p2p_list.end());
    p2p_compressed->execute(c);
#endif
  }

};

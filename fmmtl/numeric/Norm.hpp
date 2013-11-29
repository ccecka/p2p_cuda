#pragma once

#include <cmath>

#include "fmmtl/config.hpp"

template <typename T>
struct norm_type {
  typedef T type;
};

/** Compute the inner product of two doubles */
FMMTL_INLINE double inner_prod(double a, double b) {
  return a*b;
}
FMMTL_INLINE double dot(double a, double b) {
  return inner_prod(a,b);
}
/** Compute the inner product of two floats */
FMMTL_INLINE float inner_prod(float a, float b) {
  return a*b;
}
FMMTL_INLINE float dot(float a, float b) {
  return inner_prod(a,b);
}

/** Compute the squared L2 norm of a double */
FMMTL_INLINE double normSq(double a) {
  return a*a;
}
/** Compute the squared L2 norm of a float */
FMMTL_INLINE float normSq(float a) {
  return a*a;
}
/** Compute the L2 norm of a double */
FMMTL_INLINE double norm(double a) {
  return std::abs(a);
}
FMMTL_INLINE double norm_2(double a) {
  return norm(a);
}
/** Compute the L2 norm of a float */
FMMTL_INLINE float norm(float a) {
  return std::abs(a);
}
FMMTL_INLINE float norm_2(float a) {
  return norm(a);
}
/** Compute the L1 norm of a double */
FMMTL_INLINE double norm_1(double a) {
  return std::abs(a);
}
/** Compute the L1 norm of a float */
FMMTL_INLINE float norm_1(float a) {
  return std::abs(a);
}
/** Compute the L-infinity norm of a double */
FMMTL_INLINE double norm_inf(double a) {
  return std::abs(a);
}
/** Compute the L-infinity norm of a float */
FMMTL_INLINE float norm_inf(float a) {
  return std::abs(a);
}

#include "fmmtl/numeric/Complex.hpp"

using fmmtl::complex;

template <typename T>
struct norm_type<complex<T> > {
  typedef typename norm_type<T>::type type;
};

/** Compute the inner product of two complex numbers */
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type inner_prod(const complex<T>& a,
                                                 const complex<T>& b) {
  return inner_prod(a.real(),b.real()) + inner_prod(a.imag(),b.imag());
}
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type dot(const complex<T>& a,
                                          const complex<T>& b) {
  return inner_prod(a,b);
}

/** Compute the squared L2 norm of a complex */
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type normSq(const complex<T>& a) {
  return inner_prod(a.real(),a.real()) + inner_prod(a.imag(),a.imag());
}
/** Compute the L2 norm of a complex */
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type norm(const complex<T>& a) {
  using std::sqrt;
  return sqrt(normSq(a));
}
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type norm_2(const complex<T>& a) {
  return norm(a);
}
/** Compute the L1 norm of a complex */
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type norm_1(const complex<T>& a) {
  return norm_1(a.real()) + norm_1(a.imag());
}
/** Compute the L-infinity norm of a complex */
template <typename T>
FMMTL_INLINE
typename norm_type<complex<T> >::type norm_inf(const complex<T>& a) {
  using std::max;
  return max(norm_inf(a.real()), norm_inf(a.imag()));
}

#include "fmmtl/numeric/Vec.hpp"

template <std::size_t N, typename P>
struct norm_type<Vec<N,P> > {
  typedef typename norm_type<P>::type type;
};

/** Compute the inner product of two Vecs */
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type inner_prod(const Vec<N,P>& a,
                                               const Vec<N,P>& b) {
  typename norm_type<Vec<N,P> >::type v = typename norm_type<Vec<N,P> >::type();
  for (std::size_t i = 0; i != N; ++i)
    v += inner_prod(a[i],b[i]);
  return v;
}
/** Compute the dot product of two Vecs */
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type dot(const Vec<N,P>& a,
                                        const Vec<N,P>& b) {
  return inner_prod(a,b);
}

/** Compute the squared L2 norm of a Vec */
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type normSq(const Vec<N,P>& a) {
  typename norm_type<Vec<N,P> >::type v = typename norm_type<Vec<N,P> >::type();
  for (std::size_t i = 0; i != N; ++i)
    v += normSq(a[i]);
  return v;
}
/** Compute the L2 norm of a Vec */
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type norm(const Vec<N,P>& a) {
  using std::sqrt;
  return sqrt(normSq(a));
}
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type norm_2(const Vec<N,P>& a) {
  return norm(a);
}
/** Compute the L1 norm of a Vec */
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type norm_1(const Vec<N,P>& a) {
  typename norm_type<Vec<N,P> >::type v = typename norm_type<Vec<N,P> >::type();
  for (std::size_t i = 0; i != N; ++i)
    v += norm_1(a[i]);
  return v;
}
/** Compute the L-infinity norm of a Vec */
template <std::size_t N, typename P>
FMMTL_INLINE
typename norm_type<Vec<N,P> >::type norm_inf(const Vec<N,P>& a) {
  using std::max;
  typename norm_type<Vec<N,P> >::type v = typename norm_type<Vec<N,P> >::type();
  for (std::size_t i = 0; i != N; ++i)
    v = max(v, norm_inf(a[i]));
  return v;
}

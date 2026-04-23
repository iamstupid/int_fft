// int_fft — double-precision FFT bigint multiply, AVX2 + FMA required.
//
// Inputs and outputs are little-endian arrays of 64-bit limbs, interpreted as
// 4 base-2^16 "digits" per limb. The internal complex FFT uses the Bernstein
// "PQ" right-angle trick (interleaving P=even, Q=odd digits) to fit N digits
// into an N/2-point complex transform, combined with a prime-factor (Good-
// Thomas) extension that supports N = M·2^L for M ∈ {1, 3, 5, 7}.
//
// Max supported: either operand up to ~16384 limbs (65536-point FFT).
#pragma once

#include <cstddef>
#include <cstdint>

namespace fft {

// rp[0..an+bn) = ap[0..an) * bp[0..bn)
// Returns 1 on success, 0 if sizes are invalid or exceed the limit.
// rp must have room for an+bn limbs.
int mul(std::uint64_t* rp,
        const std::uint64_t* ap, std::ptrdiff_t an,
        const std::uint64_t* bp, std::ptrdiff_t bn);

// rp[0..2*an) = ap[0..an)^2
int sqr(std::uint64_t* rp, const std::uint64_t* ap, std::ptrdiff_t an);

}  // namespace fft

// -----------------------------------------------------------------------------
// Implementation. Define INT_FFT_IMPLEMENTATION in exactly one translation unit
// before including this header to pull in the function bodies.
// -----------------------------------------------------------------------------
#ifdef INT_FFT_IMPLEMENTATION

// ==================== winograd_constants.h (inlined) ========================

// --- Radix-3 (fft-pfa.md §4) ---
constexpr double W3_C              = -0.5;  // cos(2π/3) = -1/2  (exact)
constexpr double W3_S              = 0.86602540378443865;  // sin(2π/3) = √3/2

// --- Radix-5 Winograd (5 real mults, 17 real adds) ---
constexpr double W5_CP             = -0.25;  // (cos 2π/5 + cos 4π/5)/2 = -1/4 (exact)
constexpr double W5_CM             = 0.55901699437494742;  // (cos 2π/5 − cos 4π/5)/2 = √5/4
constexpr double W5_S1             = 0.95105651629515357;  // sin 2π/5
constexpr double W5_S1pS2          = 1.5388417685876267;  // sin 2π/5 + sin 4π/5
constexpr double W5_S2mS1          = -0.36327126400268044;  // sin 4π/5 − sin 2π/5

// --- Radix-7 direct form (18 real mults, ~32 adds) ---
constexpr double W7_C1             = 0.62348980185873353;  // cos 2π·1/7
constexpr double W7_C2             = -0.2225209339563144;  // cos 2π·2/7
constexpr double W7_C3             = -0.90096886790241913;  // cos 2π·3/7
constexpr double W7_S1             = 0.78183148246802981;  // sin 2π·1/7
constexpr double W7_S2             = 0.97492791218182361;  // sin 2π·2/7
constexpr double W7_S3             = 0.43388373911755812;  // sin 2π·3/7

// --- ω_M^k = exp(-2π·i·k/M) tables for PFA cross-pair pointwise dispatch ---
constexpr double W3_RE[3] = {
    1.0, -0.5, -0.5
};
constexpr double W3_IM[3] = {
    0.0, -0.86602540378443865, 0.86602540378443865
};
constexpr double W5_RE[5] = {
    1.0, 0.30901699437494742, -0.80901699437494742, -0.80901699437494742, 0.30901699437494742
};
constexpr double W5_IM[5] = {
    0.0, -0.95105651629515357, -0.58778525229247313, 0.58778525229247313, 0.95105651629515357
};
constexpr double W7_RE[7] = {
    1.0, 0.62348980185873353, -0.2225209339563144, -0.90096886790241913, -0.90096886790241913, -0.2225209339563144, 0.62348980185873353
};
constexpr double W7_IM[7] = {
    0.0, -0.78183148246802981, -0.97492791218182361, -0.43388373911755812, 0.43388373911755812, 0.97492791218182361, 0.78183148246802981
};

// ==================== fft.cpp (inlined) =====================================
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#if defined(_MSC_VER)
#  include <malloc.h>
#endif

// -----------------------------------------------------------------------------
// Anonymous namespace: everything below is internal linkage.
// -----------------------------------------------------------------------------
namespace {

#if defined(_MSC_VER)
#  define FFT_INLINE __forceinline
#else
#  define FFT_INLINE __attribute__((always_inline)) inline
#endif

// -----------------------------------------------------------------------------
// SIMD primitives: 4-lane double, 4-lane int64, and a complex-pair struct cv.
// All ops are overloaded free functions. Arithmetic operators exist for
// readability of butterflies (kept in the anon namespace to avoid leaking).
// -----------------------------------------------------------------------------

using vec4  = __m256d;
using vec4i = __m256i;

FFT_INLINE vec4 load(const double* p)        { return _mm256_load_pd(p); }
FFT_INLINE void store(double* p, vec4 x)     { _mm256_store_pd(p, x); }
FFT_INLINE vec4 zero4()                      { return _mm256_setzero_pd(); }
FFT_INLINE vec4 splat(double x)              { return _mm256_set1_pd(x); }
FFT_INLINE vec4 set_r4(double a, double b, double c, double d) {
    return _mm256_setr_pd(a, b, c, d);
}

// C++ forbids operator overloading on built-in __m256d, so vec4 uses named
// free functions. The complex-pair struct `cv` below DOES get operators.
FFT_INLINE vec4 add(vec4 a, vec4 b)           { return _mm256_add_pd(a, b); }
FFT_INLINE vec4 sub(vec4 a, vec4 b)           { return _mm256_sub_pd(a, b); }
FFT_INLINE vec4 mul(vec4 a, vec4 b)           { return _mm256_mul_pd(a, b); }
FFT_INLINE vec4 neg(vec4 a)                   { return _mm256_xor_pd(a, splat(-0.0)); }
FFT_INLINE vec4 fmadd(vec4 a, vec4 b, vec4 c) { return _mm256_fmadd_pd(a, b, c); }
FFT_INLINE vec4 fmsub(vec4 a, vec4 b, vec4 c) { return _mm256_fmsub_pd(a, b, c); }

// Reverse 4 lanes: [a,b,c,d] -> [d,c,b,a].
FFT_INLINE vec4 reverse(vec4 x) { return _mm256_permute4x64_pd(x, 0x1B); }

// Swap lanes 2 and 3: [a,b,c,d] -> [a,b,d,c]. Used for the tile-0 partner
// permutation in the PQ pointwise head (partner[0..3] = {0,1,3,2}).
FFT_INLINE vec4 swap23(vec4 x) { return _mm256_permute4x64_pd(x, 0xB4); }

// 4x4 double in-place transpose (rows <-> columns of a 4x4 block).
FFT_INLINE void transpose4(vec4& r0, vec4& r1, vec4& r2, vec4& r3) {
    vec4 t0 = _mm256_unpacklo_pd(r0, r1);
    vec4 t1 = _mm256_unpackhi_pd(r0, r1);
    vec4 t2 = _mm256_unpacklo_pd(r2, r3);
    vec4 t3 = _mm256_unpackhi_pd(r2, r3);
    r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// Complex vector: 4 complex lanes held as two vec4 (real, imag). Struct-return
// in inlined use stays in YMM registers under clang/gcc with -O3.
struct cv { vec4 re, im; };

FFT_INLINE cv load_cv(const double* p)       { return { load(p), load(p + 4) }; }
FFT_INLINE void store(double* p, cv x)       { store(p, x.re); store(p + 4, x.im); }

FFT_INLINE cv operator+(cv a, cv b)          { return { add(a.re, b.re), add(a.im, b.im) }; }
FFT_INLINE cv operator-(cv a, cv b)          { return { sub(a.re, b.re), sub(a.im, b.im) }; }
FFT_INLINE cv reverse(cv a)                  { return { reverse(a.re), reverse(a.im) }; }
FFT_INLINE cv swap23(cv a)                   { return { swap23(a.re),  swap23(a.im)  }; }

// a * w  (complex)
FFT_INLINE cv cmul(cv a, cv w) {
    return { fmsub(a.re, w.re, mul(a.im, w.im)),
             fmadd(a.re, w.im, mul(a.im, w.re)) };
}
// a * conj(w)
FFT_INLINE cv cmul_conj(cv a, cv w) {
    return { fmadd(a.im, w.im, mul(a.re, w.re)),
             fmsub(a.im, w.re, mul(a.re, w.im)) };
}
// Multiply by j*w (i.e. pre-rotate w by 90 degrees). Used in r22 odd-leg.
FFT_INLINE cv j_times(cv w) { return { neg(w.im), w.re }; }

// -----------------------------------------------------------------------------
// 128-bit lane helpers used by the final radix-2^2 tile on SSE pairs.
// -----------------------------------------------------------------------------

FFT_INLINE __m128d addsub2(__m128d x) {
    __m128d xs = _mm_shuffle_pd(x, x, 1);
    return _mm_unpacklo_pd(_mm_add_pd(x, xs), _mm_sub_pd(x, xs));
}
FFT_INLINE __m128d dup_lo(__m128d x) { return _mm_unpacklo_pd(x, x); }
FFT_INLINE __m128d dup_hi(__m128d x) { return _mm_unpackhi_pd(x, x); }

// -----------------------------------------------------------------------------
// Aligned allocator (32-byte) and grow-if-small helper.
// -----------------------------------------------------------------------------

inline void* aligned_alloc_bytes(std::size_t n) {
    if (n == 0) return nullptr;
#if defined(_MSC_VER)
    void* p = _aligned_malloc(n, 32);
    if (!p) std::abort();
#else
    void* p = nullptr;
    if (posix_memalign(&p, 32, n) != 0) std::abort();
#endif
    return p;
}
inline void aligned_release(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

template <class T>
inline void grow(T*& ptr, std::size_t& cap, std::size_t need) {
    if (cap >= need) return;
    aligned_release(ptr);
    ptr = static_cast<T*>(aligned_alloc_bytes(need * sizeof(T)));
    cap = need;
}

// -----------------------------------------------------------------------------
// Workspace and plan cache. Thread-local singletons at the bottom of the file.
// -----------------------------------------------------------------------------

struct workspace {
    double* data  = nullptr; std::size_t data_cap  = 0;
    double* data2 = nullptr; std::size_t data2_cap = 0;
};

struct plan {
    std::uint32_t max_n         = 0;
    double*       tw            = nullptr; std::size_t tw_cap            = 0;
    double*       root          = nullptr; std::size_t root_cap          = 0;
    std::uint32_t root_n        = 0;
    double*       pq_omega_br   = nullptr; std::size_t pq_omega_br_cap   = 0;
    std::uint32_t pq_omega_br_n = 0;
    // PFA factor. M == 1 -> pure power-of-two N; M ∈ {3, 5, 7} -> N = M * max_n.
    std::uint32_t M             = 1;
};

// `N_full` is the full complex FFT size (n_branch * M).
inline void ensure(workspace& ws, std::uint32_t N_full) {
    std::size_t need = 2u * std::size_t(N_full);
    grow(ws.data,  ws.data_cap,  need);
    grow(ws.data2, ws.data2_cap, need);
}

// -----------------------------------------------------------------------------
// Tiny integer helpers.
// -----------------------------------------------------------------------------

FFT_INLINE unsigned ctz_u32(std::uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    unsigned long r; _BitScanForward(&r, x); return unsigned(r);
#endif
}

inline std::uint32_t ceil_pow2(std::uint32_t x) {
    std::uint32_t r = 1;
    while (r < x) r <<= 1;
    return r;
}

FFT_INLINE std::uint32_t bitrev(std::uint32_t x, unsigned bits) {
    x = ((x & 0x55555555u) <<  1) | ((x >>  1) & 0x55555555u);
    x = ((x & 0x33333333u) <<  2) | ((x >>  2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) <<  4) | ((x >>  4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) <<  8) | ((x >>  8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return bits ? (x >> (32 - bits)) : 0u;
}

// Offset of a stage's tile-packed twiddle table inside plan.tw.
FFT_INLINE std::size_t stage_offset(unsigned lg_len) {
    assert(lg_len >= 2);
    if (lg_len == 2) return 0;
    if (lg_len == 3) return 16;
    return (std::size_t(1) << lg_len) + 16u;
}
FFT_INLINE const double* stage_tw(const plan& p, unsigned lg_len) {
    return p.tw + stage_offset(lg_len);
}
FFT_INLINE const double* stage_tw_if_present(const plan& p, unsigned lg_len) {
    return (std::uint32_t(1) << lg_len) > p.max_n ? nullptr
                                                  : stage_tw(p, lg_len);
}

// -----------------------------------------------------------------------------
// AoSoV tile access. Layout per tile of 4 complex points:
//   [re0, re1, re2, re3, im0, im1, im2, im3]  (8 doubles, 64 bytes, aligned32)
// tile index = complex_index >> 2.
// -----------------------------------------------------------------------------

FFT_INLINE       double* tile_at(      double* d, std::uint32_t i) { return d + 8u * std::size_t(i >> 2); }
FFT_INLINE const double* tile_at(const double* d, std::uint32_t i) { return d + 8u * std::size_t(i >> 2); }

FFT_INLINE double re_at(const double* d, std::uint32_t i) { return d[8u * std::size_t(i >> 2) + (i & 3u)]; }
FFT_INLINE double im_at(const double* d, std::uint32_t i) { return d[8u * std::size_t(i >> 2) + 4u + (i & 3u)]; }
FFT_INLINE void set_at(double* d, std::uint32_t i, double re, double im) {
    std::size_t off = 8u * std::size_t(i >> 2) + (i & 3u);
    d[off] = re; d[off + 4u] = im;
}

// -----------------------------------------------------------------------------
// Plan ensure: builds tile-packed twiddles and PQ right-angle roots.
// -----------------------------------------------------------------------------

inline void ensure(plan& p, std::uint32_t n) {
    if (n > p.max_n) {
        std::size_t total_tw = 0;
        for (std::uint32_t len = 4; len <= n; len <<= 1) {
            std::uint32_t l = len >> 2;
            std::size_t tile_count = (std::size_t(l) + 3u) >> 2;
            total_tw += tile_count * 16u;
        }
        grow(p.tw, p.tw_cap, total_tw);
        grow(p.root, p.root_cap, std::size_t(n));

        double* root_re = p.root;
        double* root_im = p.root + (std::size_t(n) >> 1);
        for (std::uint32_t k = 0; k < (n >> 1); ++k) {
            double ang = (2.0 * 3.14159265358979323846 * k) / double(n);
            root_re[k] = std::cos(ang);
            root_im[k] = std::sin(ang);
        }
        p.root_n = n;

        for (std::uint32_t len = 4; len <= n; len <<= 1) {
            std::uint32_t l = len >> 2;
            std::uint32_t step = n / len;
            double* dst = p.tw + stage_offset(ctz_u32(len));
            std::size_t tiles = (std::size_t(l) + 3u) >> 2;
            for (std::size_t t = 0; t < tiles; ++t) {
                std::uint32_t j = std::uint32_t(t << 2);
                for (std::uint32_t lane = 0; lane < 4; ++lane) {
                    std::uint32_t idx = j + lane;
                    if (idx < l) {
                        std::uint32_t r1 = idx * step;
                        std::uint32_t r2 = r1 << 1;
                        dst[16*t + 0 + lane] = root_re[r1];
                        dst[16*t + 4 + lane] = root_im[r1];
                        dst[16*t + 8 + lane] = root_re[r2];
                        dst[16*t +12 + lane] = root_im[r2];
                    } else {
                        dst[16*t + 0 + lane] = 1.0;
                        dst[16*t + 4 + lane] = 0.0;
                        dst[16*t + 8 + lane] = 1.0;
                        dst[16*t +12 + lane] = 0.0;
                    }
                }
            }
        }
        p.max_n = n;
    }

    grow(p.pq_omega_br, p.pq_omega_br_cap, std::size_t(p.max_n) >> 1);

    if (p.pq_omega_br_n != p.max_n) {
        const double* root_re = p.root;
        const double* root_im = p.root + (std::size_t(p.root_n) >> 1);
        std::uint32_t quarter = p.max_n >> 2;
        unsigned bits = ctz_u32(p.max_n) - 2u;
        for (std::uint32_t k = 0; k < quarter; ++k) {
            std::uint32_t e = bitrev(k, bits);
            p.pq_omega_br[k]             = root_re[e];
            p.pq_omega_br[quarter + k]   = root_im[e];
        }
        p.pq_omega_br_n = p.max_n;
    }
}

// -----------------------------------------------------------------------------
// Load 4 consecutive complex points (8 base-2^16 digits = 1 aligned uint64) as
// a cv, interpreting pairs (even-digit = real, odd-digit = imag) per the PQ
// scheme. Bounds-checked: reads at or past limb_count yield zero lanes.
// digit_idx must be a multiple of 4.
// -----------------------------------------------------------------------------

FFT_INLINE cv load_u16_pair(const std::uint64_t* src, std::size_t limb_count,
                            std::uint32_t digit_idx)
{
    const __m128i even_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                            -128, -128, -128, -128, -128, -128, -128, -128);
    const __m128i odd_mask  = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15,
                                            -128, -128, -128, -128, -128, -128, -128, -128);
    assert((digit_idx & 3u) == 0);
    std::size_t limb_idx = std::size_t(digit_idx >> 1);

    if (limb_idx >= limb_count)
        return { zero4(), zero4() };

    __m128i v;
    if (limb_idx + 1u < limb_count)
        v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + limb_idx));
    else
        v = _mm_set_epi64x(0, static_cast<long long>(src[limb_idx]));

    __m128i vr = _mm_shuffle_epi8(v, even_mask);
    __m128i vi = _mm_shuffle_epi8(v, odd_mask);
    __m256i r32 = _mm256_cvtepu16_epi32(vr);
    __m256i i32 = _mm256_cvtepu16_epi32(vi);
    cv out;
    out.re = _mm256_cvtepi32_pd(_mm256_castsi256_si128(r32));
    out.im = _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32));
    return out;
}

// -----------------------------------------------------------------------------
// Final radix-2^2 tile: an in-tile 4-point DFT, done in SSE pairs.
// -----------------------------------------------------------------------------

FFT_INLINE void fwd_tile(double* p) {
    const __m128d sign = _mm_set1_pd(-0.0);
    __m128d r01 = _mm_loadu_pd(p + 0);
    __m128d r23 = _mm_loadu_pd(p + 2);
    __m128d i01 = _mm_loadu_pd(p + 4);
    __m128d i23 = _mm_loadu_pd(p + 6);

    __m128d sumr  = _mm_add_pd(r01, r23);
    __m128d sumi  = _mm_add_pd(i01, i23);
    __m128d diffr = _mm_sub_pd(r01, r23);
    __m128d diffi = _mm_sub_pd(i01, i23);

    __m128d o01r = addsub2(sumr);
    __m128d o01i = addsub2(sumi);
    __m128d o23r = addsub2(_mm_unpacklo_pd(dup_lo(diffr),
                                           _mm_xor_pd(dup_hi(diffi), sign)));
    __m128d o23i = addsub2(_mm_unpacklo_pd(dup_lo(diffi), dup_hi(diffr)));

    _mm_storeu_pd(p + 0, o01r);
    _mm_storeu_pd(p + 2, o23r);
    _mm_storeu_pd(p + 4, o01i);
    _mm_storeu_pd(p + 6, o23i);
}

FFT_INLINE void inv_tile(double* p) {
    const __m128d sign = _mm_set1_pd(-0.0);
    __m128d o01r = _mm_loadu_pd(p + 0);
    __m128d o23r = _mm_loadu_pd(p + 2);
    __m128d o01i = _mm_loadu_pd(p + 4);
    __m128d o23i = _mm_loadu_pd(p + 6);

    __m128d e01r = addsub2(o01r);
    __m128d e01i = addsub2(o01i);
    __m128d e23r = addsub2(o23r);
    __m128d e23i = addsub2(o23i);

    __m128d x0r = _mm_add_pd(dup_lo(e01r), dup_lo(e23r));
    __m128d x0i = _mm_add_pd(dup_lo(e01i), dup_lo(e23i));
    __m128d x2r = _mm_sub_pd(dup_lo(e01r), dup_lo(e23r));
    __m128d x2i = _mm_sub_pd(dup_lo(e01i), dup_lo(e23i));
    __m128d x1r = _mm_add_pd(dup_hi(e01r), dup_hi(e23i));
    __m128d x1i = _mm_add_pd(dup_hi(e01i), _mm_xor_pd(dup_hi(e23r), sign));
    __m128d x3r = _mm_sub_pd(dup_hi(e01r), dup_hi(e23i));
    __m128d x3i = _mm_sub_pd(dup_hi(e01i), _mm_xor_pd(dup_hi(e23r), sign));

    _mm_storeu_pd(p + 0, _mm_unpacklo_pd(x0r, x1r));
    _mm_storeu_pd(p + 2, _mm_unpacklo_pd(x2r, x3r));
    _mm_storeu_pd(p + 4, _mm_unpacklo_pd(x0i, x1i));
    _mm_storeu_pd(p + 6, _mm_unpacklo_pd(x2i, x3i));
}

// Same kernel as fwd_tile but the inputs are supplied as a cv.
FFT_INLINE void fwd_pair(double* p, cv x) {
    const __m128d sign = _mm_set1_pd(-0.0);
    __m128d r01 = _mm256_castpd256_pd128(x.re);
    __m128d r23 = _mm256_extractf128_pd(x.re, 1);
    __m128d i01 = _mm256_castpd256_pd128(x.im);
    __m128d i23 = _mm256_extractf128_pd(x.im, 1);

    __m128d sumr  = _mm_add_pd(r01, r23);
    __m128d sumi  = _mm_add_pd(i01, i23);
    __m128d diffr = _mm_sub_pd(r01, r23);
    __m128d diffi = _mm_sub_pd(i01, i23);

    __m128d o01r = addsub2(sumr);
    __m128d o01i = addsub2(sumi);
    __m128d o23r = addsub2(_mm_unpacklo_pd(dup_lo(diffr),
                                           _mm_xor_pd(dup_hi(diffi), sign)));
    __m128d o23i = addsub2(_mm_unpacklo_pd(dup_lo(diffi), dup_hi(diffr)));

    _mm_storeu_pd(p + 0, o01r);
    _mm_storeu_pd(p + 2, o23r);
    _mm_storeu_pd(p + 4, o01i);
    _mm_storeu_pd(p + 6, o23i);
}

FFT_INLINE cv inv_pair(cv x) {
    const __m128d sign = _mm_set1_pd(-0.0);
    __m128d o01r = _mm256_castpd256_pd128(x.re);
    __m128d o23r = _mm256_extractf128_pd(x.re, 1);
    __m128d o01i = _mm256_castpd256_pd128(x.im);
    __m128d o23i = _mm256_extractf128_pd(x.im, 1);

    __m128d e01r = addsub2(o01r);
    __m128d e01i = addsub2(o01i);
    __m128d e23r = addsub2(o23r);
    __m128d e23i = addsub2(o23i);

    __m128d x0r = _mm_add_pd(dup_lo(e01r), dup_lo(e23r));
    __m128d x0i = _mm_add_pd(dup_lo(e01i), dup_lo(e23i));
    __m128d x2r = _mm_sub_pd(dup_lo(e01r), dup_lo(e23r));
    __m128d x2i = _mm_sub_pd(dup_lo(e01i), dup_lo(e23i));
    __m128d x1r = _mm_add_pd(dup_hi(e01r), dup_hi(e23i));
    __m128d x1i = _mm_add_pd(dup_hi(e01i), _mm_xor_pd(dup_hi(e23r), sign));
    __m128d x3r = _mm_sub_pd(dup_hi(e01r), dup_hi(e23i));
    __m128d x3i = _mm_sub_pd(dup_hi(e01i), _mm_xor_pd(dup_hi(e23r), sign));

    __m128d rlo = _mm_unpacklo_pd(x0r, x1r);
    __m128d rhi = _mm_unpacklo_pd(x2r, x3r);
    __m128d ilo = _mm_unpacklo_pd(x0i, x1i);
    __m128d ihi = _mm_unpacklo_pd(x2i, x3i);

    cv out;
    out.re = _mm256_insertf128_pd(_mm256_castpd128_pd256(rlo), rhi, 1);
    out.im = _mm256_insertf128_pd(_mm256_castpd128_pd256(ilo), ihi, 1);
    return out;
}

// 4 final tiles batched: transpose 4x4, do 4 parallel radix-2^2 tiles, transpose back.
FFT_INLINE void fwd_pair4(double* p, cv a0, cv a1, cv a2, cv a3) {
    transpose4(a0.re, a1.re, a2.re, a3.re);
    transpose4(a0.im, a1.im, a2.im, a3.im);

    cv e0 = a0 + a2, e1 = a0 - a2;
    cv o0 = a1 + a3, o1 = a1 - a3;

    cv y0 = e0 + o0;
    cv y1 = { sub(e1.re, o1.im), add(e1.im, o1.re) };
    cv y2 = e0 - o0;
    cv y3 = { add(e1.re, o1.im), sub(e1.im, o1.re) };

    // store order (y0, y2, y1, y3) matches the radix-2^2 bit-reversal.
    transpose4(y0.re, y2.re, y1.re, y3.re);
    transpose4(y0.im, y2.im, y1.im, y3.im);

    store(p + 0,  y0);
    store(p + 8,  y2);
    store(p + 16, y1);
    store(p + 24, y3);
}

FFT_INLINE void inv_pair4(cv& o0, cv& o1, cv& o2, cv& o3,
                          cv y0, cv y1, cv y2, cv y3) {
    transpose4(y0.re, y1.re, y2.re, y3.re);
    transpose4(y0.im, y1.im, y2.im, y3.im);

    cv a0 = y0 + y1, a1 = y0 - y1;
    cv b0 = y2 + y3, b1 = y2 - y3;

    cv x0 = a0 + b0;
    cv x1 = { add(a1.re, b1.im), sub(a1.im, b1.re) };
    cv x2 = a0 - b0;
    cv x3 = { sub(a1.re, b1.im), add(a1.im, b1.re) };

    transpose4(x0.re, x1.re, x2.re, x3.re);
    transpose4(x0.im, x1.im, x2.im, x3.im);
    o0 = x0; o1 = x1; o2 = x2; o3 = x3;
}

// Batch variant: load 4 tiles, do the tiles' DFTs in parallel, store back.
FFT_INLINE void fwd_tile4(double* p) {
    cv a0 = load_cv(p + 0);
    cv a1 = load_cv(p + 8);
    cv a2 = load_cv(p + 16);
    cv a3 = load_cv(p + 24);

    transpose4(a0.re, a1.re, a2.re, a3.re);
    transpose4(a0.im, a1.im, a2.im, a3.im);

    cv e0 = a0 + a2, e1 = a0 - a2;
    cv o0 = a1 + a3, o1 = a1 - a3;

    cv y0 = e0 + o0;
    cv y1 = { sub(e1.re, o1.im), add(e1.im, o1.re) };
    cv y2 = e0 - o0;
    cv y3 = { add(e1.re, o1.im), sub(e1.im, o1.re) };

    // note: y0, y2, y1, y3 interleave order for final stride.
    transpose4(y0.re, y2.re, y1.re, y3.re);
    transpose4(y0.im, y2.im, y1.im, y3.im);

    store(p + 0,  y0);
    store(p + 8,  y2);
    store(p + 16, y1);
    store(p + 24, y3);
}

FFT_INLINE void inv_tile4(double* p) {
    cv y0 = load_cv(p + 0);
    cv y1 = load_cv(p + 8);
    cv y2 = load_cv(p + 16);
    cv y3 = load_cv(p + 24);

    // Fold the y1<->y2 swap into the transpose by permuting arg order:
    // transpose4(a,b,c,d) writes to a,b,c,d; passing (y0,y2,y1,y3) directs
    // the row-1 result into y2 and the row-2 result into y1.
    transpose4(y0.re, y2.re, y1.re, y3.re);
    transpose4(y0.im, y2.im, y1.im, y3.im);

    cv a0 = y0 + y2, a1 = y0 - y2;
    cv b0 = y1 + y3, b1 = y1 - y3;

    cv x0 = a0 + b0;
    cv x1 = { add(a1.re, b1.im), sub(a1.im, b1.re) };
    cv x2 = a0 - b0;
    cv x3 = { sub(a1.re, b1.im), add(a1.im, b1.re) };

    transpose4(x0.re, x1.re, x2.re, x3.re);
    transpose4(x0.im, x1.im, x2.im, x3.im);

    store(p + 0,  x0);
    store(p + 8,  x1);
    store(p + 16, x2);
    store(p + 24, x3);
}

// -----------------------------------------------------------------------------
// Final radix-2^3 blocks: 8-point butterfly collapsed with a twiddle-by-root-8
// then dispatched into fwd_pair (single) or fwd_pair4 (batch of 2).
// -----------------------------------------------------------------------------

FFT_INLINE void fwd_tail8(double* p) {
    constexpr double c = 0.70710678118654752440;
    cv w = { set_r4(1.0, c, 0.0, -c),
             set_r4(0.0, c, 1.0,  c) };
    cv a = load_cv(p + 0);
    cv b = load_cv(p + 8);
    cv s = a + b;
    cv d = cmul(a - b, w);
    fwd_pair(p,     s);
    fwd_pair(p + 8, d);
}

FFT_INLINE void inv_tail8(double* p) {
    constexpr double c = 0.70710678118654752440;
    cv w = { set_r4(1.0, c, 0.0, -c),
             set_r4(0.0, c, 1.0,  c) };
    cv a = load_cv(p + 0);
    cv b = load_cv(p + 8);
    a = inv_pair(a);
    b = inv_pair(b);
    cv t = cmul_conj(b, w);
    store(p + 0, a + t);
    store(p + 8, a - t);
}

FFT_INLINE void fwd_tail8x2(double* p) {
    constexpr double c = 0.70710678118654752440;
    cv w = { set_r4(1.0, c, 0.0, -c),
             set_r4(0.0, c, 1.0,  c) };

    cv a0 = load_cv(p + 0);
    cv b0 = load_cv(p + 8);
    cv a1 = load_cv(p + 16);
    cv b1 = load_cv(p + 24);

    cv s0 = a0 + b0;
    cv d0 = cmul(a0 - b0, w);
    cv s1 = a1 + b1;
    cv d1 = cmul(a1 - b1, w);

    fwd_pair4(p, s0, d0, s1, d1);
}

FFT_INLINE void inv_tail8x2(double* p) {
    constexpr double c = 0.70710678118654752440;
    cv w = { set_r4(1.0, c, 0.0, -c),
             set_r4(0.0, c, 1.0,  c) };

    cv s0 = load_cv(p + 0);
    cv d0 = load_cv(p + 8);
    cv s1 = load_cv(p + 16);
    cv d1 = load_cv(p + 24);

    inv_pair4(s0, d0, s1, d1, s0, d0, s1, d1);

    cv t0 = cmul_conj(d0, w);
    store(p + 0,  s0 + t0);
    store(p + 8,  s0 - t0);
    cv t1 = cmul_conj(d1, w);
    store(p + 16, s1 + t1);
    store(p + 24, s1 - t1);
}

// -----------------------------------------------------------------------------
// 16-complex tail block: one radix-2^2 butterfly followed by fwd_pair4 (or
// the inverse analogue).
// -----------------------------------------------------------------------------

FFT_INLINE void fwd_tail16(const double* tw, double* p) {
    cv a0 = load_cv(p + 0);
    cv a1 = load_cv(p + 8);
    cv a2 = load_cv(p + 16);
    cv a3 = load_cv(p + 24);

    cv w1 = { load(tw + 0), load(tw + 4)  };
    cv w2 = { load(tw + 8), load(tw + 12) };

    cv b0 = a0 + a2;
    cv b2 = cmul(a0 - a2, w1);
    cv b1 = a1 + a3;
    cv b3 = cmul(a1 - a3, j_times(w1));

    cv c0 = b0 + b1;
    cv c1 = cmul(b0 - b1, w2);
    cv c2 = b2 + b3;
    cv c3 = cmul(b2 - b3, w2);

    fwd_pair4(p, c0, c1, c2, c3);
}

FFT_INLINE void inv_tail16(const double* tw, double* p) {
    cv y0 = load_cv(p + 0);
    cv y1 = load_cv(p + 8);
    cv y2 = load_cv(p + 16);
    cv y3 = load_cv(p + 24);

    inv_pair4(y0, y1, y2, y3, y0, y1, y2, y3);

    cv w1 = { load(tw + 0), load(tw + 4)  };
    cv w2 = { load(tw + 8), load(tw + 12) };

    cv t  = cmul_conj(y1, w2);
    cv b0 = y0 + t,  b1 = y0 - t;
    t     = cmul_conj(y3, w2);
    cv b2 = y2 + t,  b3 = y2 - t;

    t = cmul_conj(b2, w1);
    store(p + 0,  b0 + t);
    store(p + 16, b0 - t);
    t = cmul_conj(b3, j_times(w1));
    store(p + 8,  b1 + t);
    store(p + 24, b1 - t);
}

// -----------------------------------------------------------------------------
// Radix-2^2 butterfly (DIF for forward, DIT for inverse).
// Operates on 4 tiles of 4 complex lanes each = 16 points.
// w1 = twiddle for half-stage, w2 = twiddle for full stage.
// -----------------------------------------------------------------------------

FFT_INLINE void r22_dif(double* d0, double* d1, double* d2, double* d3,
                        const double* s0, const double* s1,
                        const double* s2, const double* s3,
                        cv w1, cv w2)
{
    cv a0 = load_cv(s0);
    cv a1 = load_cv(s1);
    cv a2 = load_cv(s2);
    cv a3 = load_cv(s3);

    cv b0 = a0 + a2;
    cv b2 = cmul(a0 - a2, w1);
    cv b1 = a1 + a3;
    cv b3 = cmul(a1 - a3, j_times(w1));

    store(d0, b0 + b1);
    store(d1, cmul(b0 - b1, w2));
    store(d2, b2 + b3);
    store(d3, cmul(b2 - b3, w2));
}

FFT_INLINE void r22_dit(double* d0, double* d1, double* d2, double* d3,
                        const double* s0, const double* s1,
                        const double* s2, const double* s3,
                        cv w1, cv w2)
{
    cv y0 = load_cv(s0);
    cv y1 = load_cv(s1);
    cv t  = cmul_conj(y1, w2);
    cv b0 = y0 + t, b1 = y0 - t;

    cv y2 = load_cv(s2);
    cv y3 = load_cv(s3);
    t     = cmul_conj(y3, w2);
    cv b2 = y2 + t, b3 = y2 - t;

    t = cmul_conj(b2, w1);
    store(d0, b0 + t);
    store(d2, b0 - t);

    t = cmul_conj(b3, j_times(w1));
    store(d1, b1 + t);
    store(d3, b1 - t);
}

// -----------------------------------------------------------------------------
// Scalar radix-2 stage for n == 2 fallback (forward == inverse, no twiddles).
// -----------------------------------------------------------------------------

inline void r2_stage(double* d, std::uint32_t n) {
    for (std::uint32_t b = 0; b < n; b += 2) {
        double ar = re_at(d, b), ai = im_at(d, b);
        double br = re_at(d, b + 1), bi = im_at(d, b + 1);
        set_at(d, b,     ar + br, ai + bi);
        set_at(d, b + 1, ar - br, ai - bi);
    }
}

// Tail block size (in complex points) depending on log2(n) parity.
FFT_INLINE std::uint32_t tail_block_complex(std::uint32_t n) {
    assert((n & (n - 1u)) == 0);
    if (n >= 16u && ((ctz_u32(n) & 1u) == 0)) return 16u;
    if (n >= 8u) return 8u;
    return 4u;
}

// -----------------------------------------------------------------------------
// Tail range: runs the final stages over [start, stop). Picks tile size
// per tail_block_complex(stop - start).
// -----------------------------------------------------------------------------

inline void fwd_tail_range(double* d, std::uint32_t start, std::uint32_t stop,
                           const double* tw16)
{
    std::uint32_t blk = tail_block_complex(stop - start);
    std::uint32_t t;

    if (blk == 16u) {
        for (t = start; t < stop; t += 16u)
            fwd_tail16(tw16, tile_at(d, t));
        return;
    }
    if (blk == 8u) {
        for (t = start; t + 16u <= stop; t += 16u)
            fwd_tail8x2(tile_at(d, t));
        for (; t < stop; t += 8u)
            fwd_tail8(tile_at(d, t));
        return;
    }
    for (t = start; t + 16u <= stop; t += 16u)
        fwd_tile4(tile_at(d, t));
    for (; t < stop; t += 4u)
        fwd_tile(tile_at(d, t));
}

inline void inv_tail_range(double* d, std::uint32_t start, std::uint32_t stop,
                           const double* tw16)
{
    std::uint32_t blk = tail_block_complex(stop - start);
    std::uint32_t t;

    if (blk == 16u) {
        for (t = start; t < stop; t += 16u)
            inv_tail16(tw16, tile_at(d, t));
        return;
    }
    if (blk == 8u) {
        for (t = start; t + 16u <= stop; t += 16u)
            inv_tail8x2(tile_at(d, t));
        for (; t < stop; t += 8u)
            inv_tail8(tile_at(d, t));
        return;
    }
    for (t = start; t + 16u <= stop; t += 16u)
        inv_tile4(tile_at(d, t));
    for (; t < stop; t += 4u)
        inv_tile(tile_at(d, t));
}

// -----------------------------------------------------------------------------
// One full radix-2^2 pass over [start, stop) at length `len`.
// -----------------------------------------------------------------------------

inline void fwd_range(double* d, std::uint32_t start, std::uint32_t stop,
                      std::uint32_t len, const double* tw)
{
    std::uint32_t l = len >> 2;
    std::size_t tile_stride = 2u * std::size_t(l);
    assert((l & 3u) == 0);

    for (std::uint32_t base = start; base < stop; base += len) {
        const double* twp = tw;
        std::uint32_t blocks = l >> 2;
        double* p0 = tile_at(d, base);
        double* p1 = p0 + tile_stride;
        double* p2 = p1 + tile_stride;
        double* p3 = p2 + tile_stride;

        while (blocks >= 2u) {
            cv w1a = { load(twp +  0), load(twp +  4) };
            cv w2a = { load(twp +  8), load(twp + 12) };
            cv w1b = { load(twp + 16), load(twp + 20) };
            cv w2b = { load(twp + 24), load(twp + 28) };
            r22_dif(p0,     p1,     p2,     p3,
                    p0,     p1,     p2,     p3,     w1a, w2a);
            r22_dif(p0 + 8, p1 + 8, p2 + 8, p3 + 8,
                    p0 + 8, p1 + 8, p2 + 8, p3 + 8, w1b, w2b);
            p0 += 16; p1 += 16; p2 += 16; p3 += 16;
            twp += 32;
            blocks -= 2u;
        }
        while (blocks-- != 0u) {
            cv w1 = { load(twp + 0), load(twp + 4)  };
            cv w2 = { load(twp + 8), load(twp + 12) };
            r22_dif(p0, p1, p2, p3, p0, p1, p2, p3, w1, w2);
            p0 += 8; p1 += 8; p2 += 8; p3 += 8;
            twp += 16;
        }
    }
}

inline void inv_range(double* d, std::uint32_t start, std::uint32_t stop,
                      std::uint32_t len, const double* tw)
{
    std::uint32_t l = len >> 2;
    std::size_t tile_stride = 2u * std::size_t(l);
    assert((l & 3u) == 0);

    for (std::uint32_t base = start; base < stop; base += len) {
        const double* twp = tw;
        std::uint32_t blocks = l >> 2;
        double* p0 = tile_at(d, base);
        double* p1 = p0 + tile_stride;
        double* p2 = p1 + tile_stride;
        double* p3 = p2 + tile_stride;

        while (blocks >= 2u) {
            cv w1a = { load(twp +  0), load(twp +  4) };
            cv w2a = { load(twp +  8), load(twp + 12) };
            cv w1b = { load(twp + 16), load(twp + 20) };
            cv w2b = { load(twp + 24), load(twp + 28) };
            r22_dit(p0,     p1,     p2,     p3,
                    p0,     p1,     p2,     p3,     w1a, w2a);
            r22_dit(p0 + 8, p1 + 8, p2 + 8, p3 + 8,
                    p0 + 8, p1 + 8, p2 + 8, p3 + 8, w1b, w2b);
            p0 += 16; p1 += 16; p2 += 16; p3 += 16;
            twp += 32;
            blocks -= 2u;
        }
        while (blocks-- != 0u) {
            cv w1 = { load(twp + 0), load(twp + 4)  };
            cv w2 = { load(twp + 8), load(twp + 12) };
            r22_dit(p0, p1, p2, p3, p0, p1, p2, p3, w1, w2);
            p0 += 8; p1 += 8; p2 += 8; p3 += 8;
            twp += 16;
        }
    }
}

// -----------------------------------------------------------------------------
// Unpack + first-stage fusion: reads u16 digits directly out of the input
// limbs, applies the first radix-2^2 butterfly, and writes tiles to data.
// Used for n >= 32.
// -----------------------------------------------------------------------------

inline void unpack_fwd(double* data, std::uint32_t n,
                       const std::uint64_t* src, std::size_t limb_count,
                       const double* tw)
{
    std::uint32_t l = n >> 2;
    assert((l & 3u) == 0);

    double* p0 = tile_at(data, 0);
    double* p1 = tile_at(data, l);
    double* p2 = tile_at(data, 2 * l);
    double* p3 = tile_at(data, 3 * l);

    for (std::uint32_t j = 0; j < l; j += 4u) {
        cv a0 = load_u16_pair(src, limb_count, j + 0u * l);
        cv a1 = load_u16_pair(src, limb_count, j + 1u * l);
        cv a2 = load_u16_pair(src, limb_count, j + 2u * l);
        cv a3 = load_u16_pair(src, limb_count, j + 3u * l);

        cv w1 = { load(tw + 0), load(tw + 4)  };
        cv w2 = { load(tw + 8), load(tw + 12) };

        cv b0 = a0 + a2;
        cv b2 = cmul(a0 - a2, w1);
        cv b1 = a1 + a3;
        cv b3 = cmul(a1 - a3, j_times(w1));

        store(p0, b0 + b1);
        store(p1, cmul(b0 - b1, w2));
        store(p2, b2 + b3);
        store(p3, cmul(b2 - b3, w2));

        p0 += 8; p1 += 8; p2 += 8; p3 += 8;
        tw += 16;
    }
}

// -----------------------------------------------------------------------------
// Prime-factor (Good-Thomas) PFA input/output stages for M ∈ {3, 5, 7}.
//
// For N_full = M * n, split the natural-order input into M branches stored at
// offsets {0, 2n, 4n, ..., 2n(M-1)} doubles in `data`. Branch b holds n
// complex points at PFA-input coords (b = k mod M, k1 = k mod n).
//
// Per iteration we read M stripes at complex offsets that are (cyclic)
// rotations of {0, n, 2n, ..., (M-1)n}, lane-blend them into M per-a vec4s,
// apply a radix-M butterfly, and write M branch tiles.
//
// The per-iteration phase (lane-0 a-residue bump of 4 mod M) is absorbed
// into rotating stripe-offset labels, so the shuffle step uses fixed
// per-output blend4 patterns with no run-time branching. Offsets are held
// in a single __m256i (8×u32 slots, enough for M ≤ 8): per-iter advance is
// one vpaddd, cyclic relabeling is one vpermd with an M-specific rotation
// mask. See fft-pfa-57.md §§4–6 for the derivation.
// -----------------------------------------------------------------------------

// Generic 4-lane blend: emit (a.lane0, b.lane1, c.lane2, d.lane3) per re/im.
// 3 blend_pd per re, 3 per im — 6 blend ops total. Used in both the forward
// shuffle-fwd (x_A lane l = s_{(A−l) mod M}) and the inverse shuffle-out
// (phi-slot p lane l = y_{(p+l) mod M}) with callers supplying the 4 args in
// the right cyclic order. For M=3, lane 3 is a repeat of lane 0 (d == a);
// the extra blend is negligible (see §4 of the design doc).
FFT_INLINE cv blend4(cv a, cv b, cv c, cv d) {
    vec4 re_ab  = _mm256_blend_pd(a.re, b.re, 0b0010);
    vec4 re_abc = _mm256_blend_pd(re_ab, c.re, 0b0100);
    vec4 re     = _mm256_blend_pd(re_abc, d.re, 0b1000);
    vec4 im_ab  = _mm256_blend_pd(a.im, b.im, 0b0010);
    vec4 im_abc = _mm256_blend_pd(im_ab, c.im, 0b0100);
    vec4 im     = _mm256_blend_pd(im_abc, d.im, 0b1000);
    return { re, im };
}

// -- Radix-3 Winograd butterfly (3 real mults, 6 adds, Rader-Brenner form) --

FFT_INLINE void pfa3_butterfly_fwd(cv x0, cv x1, cv x2,
                                   cv& y0, cv& y1, cv& y2)
{
    const vec4 neg_half = splat(W3_C);
    const vec4 sqrt3_2  = splat(W3_S);

    cv s = x1 + x2;
    cv d = x1 - x2;
    y0 = x0 + s;
    cv u  = { fmadd(neg_half, s.re, x0.re),
              fmadd(neg_half, s.im, x0.im) };
    cv v  = { mul(sqrt3_2, d.re),
              mul(sqrt3_2, d.im) };
    y1 = { add(u.re, v.im), sub(u.im, v.re) };   // u - i*v
    y2 = { sub(u.re, v.im), add(u.im, v.re) };   // u + i*v
}

FFT_INLINE void pfa3_butterfly_inv(cv x0, cv x1, cv x2,
                                   cv& y0, cv& y1, cv& y2)
{
    const vec4 neg_half = splat(W3_C);
    const vec4 sqrt3_2  = splat(W3_S);

    cv s = x1 + x2;
    cv d = x1 - x2;
    y0 = x0 + s;
    cv u  = { fmadd(neg_half, s.re, x0.re),
              fmadd(neg_half, s.im, x0.im) };
    cv v  = { mul(sqrt3_2, d.re),
              mul(sqrt3_2, d.im) };
    y1 = { sub(u.re, v.im), add(u.im, v.re) };   // u + i*v
    y2 = { add(u.re, v.im), sub(u.im, v.re) };   // u - i*v
}

// -- Radix-5 Winograd butterfly (5 complex×real mults, 17 adds) --
// Shared core for fwd/inv; only the final ±i·I sign on outputs differs.

FFT_INLINE void pfa5_butterfly_fwd(cv x0, cv x1, cv x2, cv x3, cv x4,
                                   cv& y0, cv& y1, cv& y2, cv& y3, cv& y4)
{
    const vec4 CP  = splat(W5_CP);
    const vec4 CM  = splat(W5_CM);
    const vec4 S1  = splat(W5_S1);
    const vec4 S1p = splat(W5_S1pS2);
    const vec4 S2m = splat(W5_S2mS1);

    cv u14 = x1 + x4, v14 = x1 - x4;
    cv u25 = x2 + x3, v25 = x2 - x3;
    cv us  = u14 + u25;
    cv um  = u14 - u25;
    cv vs  = v14 - v25;

    cv m1  = { mul(us.re,  CP),  mul(us.im,  CP)  };
    cv m2  = { mul(um.re,  CM),  mul(um.im,  CM)  };
    cv m3  = { mul(vs.re,  S1),  mul(vs.im,  S1)  };
    cv m4  = { mul(v25.re, S1p), mul(v25.im, S1p) };
    cv m5  = { mul(v14.re, S2m), mul(v14.im, S2m) };

    cv I14 = m3 + m4;       // = s1·v14 + s2·v25
    cv I23 = m3 + m5;       // = s2·v14 − s1·v25
    cv mid = x0 + m1;       // = x0 − us/4
    cv R14 = mid + m2;
    cv R23 = mid - m2;

    y0 = x0 + us;
    // Y_k = R − i·I (k=1,2) / R + i·I (k=3,4). (−i·z).re = z.im, (−i·z).im = −z.re.
    y1 = { add(R14.re, I14.im), sub(R14.im, I14.re) };
    y4 = { sub(R14.re, I14.im), add(R14.im, I14.re) };
    y2 = { add(R23.re, I23.im), sub(R23.im, I23.re) };
    y3 = { sub(R23.re, I23.im), add(R23.im, I23.re) };
}

FFT_INLINE void pfa5_butterfly_inv(cv x0, cv x1, cv x2, cv x3, cv x4,
                                   cv& y0, cv& y1, cv& y2, cv& y3, cv& y4)
{
    const vec4 CP  = splat(W5_CP);
    const vec4 CM  = splat(W5_CM);
    const vec4 S1  = splat(W5_S1);
    const vec4 S1p = splat(W5_S1pS2);
    const vec4 S2m = splat(W5_S2mS1);

    cv u14 = x1 + x4, v14 = x1 - x4;
    cv u25 = x2 + x3, v25 = x2 - x3;
    cv us  = u14 + u25;
    cv um  = u14 - u25;
    cv vs  = v14 - v25;

    cv m1  = { mul(us.re,  CP),  mul(us.im,  CP)  };
    cv m2  = { mul(um.re,  CM),  mul(um.im,  CM)  };
    cv m3  = { mul(vs.re,  S1),  mul(vs.im,  S1)  };
    cv m4  = { mul(v25.re, S1p), mul(v25.im, S1p) };
    cv m5  = { mul(v14.re, S2m), mul(v14.im, S2m) };

    cv I14 = m3 + m4;
    cv I23 = m3 + m5;
    cv mid = x0 + m1;
    cv R14 = mid + m2;
    cv R23 = mid - m2;

    y0 = x0 + us;
    // Inverse: swap ±i sign on every output (equivalent to y1↔y4, y2↔y3).
    y1 = { sub(R14.re, I14.im), add(R14.im, I14.re) };
    y4 = { add(R14.re, I14.im), sub(R14.im, I14.re) };
    y2 = { sub(R23.re, I23.im), add(R23.im, I23.re) };
    y3 = { add(R23.re, I23.im), sub(R23.im, I23.re) };
}

// -- Radix-7 direct-form butterfly (18 real mults / 9 complex×real, ~32 adds) --
// Each R_k is x0 + <cyclic-shifted (c1,c2,c3)> · (u16,u25,u34);
// each I_k is <cyclic-shifted, signed (s1,s2,s3)> · (v16,v25,v34).
// Implemented as chains of fmadd/fmsub (3-term sums → 2 FMAs + 1 mul/fmadd).
// A Winograd-7 (8 non-trivial mults) was prototyped but ran ~5% slower on
// Zen4 AVX2+FMA — the extra adds contend with FMAs for the 2 FP ports, and
// FMA makes the mult-count savings moot. See winograd_constants.py for the
// validated Rader + 2×3 Agarwal-Cooley factorization if that tradeoff ever
// flips on a future microarchitecture.

FFT_INLINE void pfa7_butterfly_fwd(cv x0, cv x1, cv x2, cv x3, cv x4, cv x5, cv x6,
                                   cv& y0, cv& y1, cv& y2, cv& y3, cv& y4, cv& y5, cv& y6)
{
    const vec4 C1 = splat(W7_C1), C2 = splat(W7_C2), C3 = splat(W7_C3);
    const vec4 S1 = splat(W7_S1), S2 = splat(W7_S2), S3 = splat(W7_S3);

    cv u16 = x1 + x6, v16 = x1 - x6;
    cv u25 = x2 + x5, v25 = x2 - x5;
    cv u34 = x3 + x4, v34 = x3 - x4;

    cv R1, R2, R3, I1, I2, I3;

    // R_k = x0 + c_{(k·1) mod7-wrap} u16 + c_{(k·2)} u25 + c_{(k·3)} u34
    R1.re = fmadd(C3, u34.re, fmadd(C2, u25.re, fmadd(C1, u16.re, x0.re)));
    R1.im = fmadd(C3, u34.im, fmadd(C2, u25.im, fmadd(C1, u16.im, x0.im)));
    R2.re = fmadd(C1, u34.re, fmadd(C3, u25.re, fmadd(C2, u16.re, x0.re)));
    R2.im = fmadd(C1, u34.im, fmadd(C3, u25.im, fmadd(C2, u16.im, x0.im)));
    R3.re = fmadd(C2, u34.re, fmadd(C1, u25.re, fmadd(C3, u16.re, x0.re)));
    R3.im = fmadd(C2, u34.im, fmadd(C1, u25.im, fmadd(C3, u16.im, x0.im)));

    // I_1 =  s1·v16 + s2·v25 + s3·v34
    I1.re = fmadd(S3, v34.re, fmadd(S2, v25.re, mul(S1, v16.re)));
    I1.im = fmadd(S3, v34.im, fmadd(S2, v25.im, mul(S1, v16.im)));
    // I_2 =  s2·v16 − s3·v25 − s1·v34 = s2·v16 − (s1·v34 + s3·v25)
    I2.re = fmsub(S2, v16.re, fmadd(S1, v34.re, mul(S3, v25.re)));
    I2.im = fmsub(S2, v16.im, fmadd(S1, v34.im, mul(S3, v25.im)));
    // I_3 =  s3·v16 − s1·v25 + s2·v34 = s3·v16 + (s2·v34 − s1·v25)
    I3.re = fmadd(S3, v16.re, fmsub(S2, v34.re, mul(S1, v25.re)));
    I3.im = fmadd(S3, v16.im, fmsub(S2, v34.im, mul(S1, v25.im)));

    y0 = { add(add(x0.re, u16.re), add(u25.re, u34.re)),
           add(add(x0.im, u16.im), add(u25.im, u34.im)) };

    y1 = { add(R1.re, I1.im), sub(R1.im, I1.re) };
    y6 = { sub(R1.re, I1.im), add(R1.im, I1.re) };
    y2 = { add(R2.re, I2.im), sub(R2.im, I2.re) };
    y5 = { sub(R2.re, I2.im), add(R2.im, I2.re) };
    y3 = { add(R3.re, I3.im), sub(R3.im, I3.re) };
    y4 = { sub(R3.re, I3.im), add(R3.im, I3.re) };
}

FFT_INLINE void pfa7_butterfly_inv(cv x0, cv x1, cv x2, cv x3, cv x4, cv x5, cv x6,
                                   cv& y0, cv& y1, cv& y2, cv& y3, cv& y4, cv& y5, cv& y6)
{
    const vec4 C1 = splat(W7_C1), C2 = splat(W7_C2), C3 = splat(W7_C3);
    const vec4 S1 = splat(W7_S1), S2 = splat(W7_S2), S3 = splat(W7_S3);

    cv u16 = x1 + x6, v16 = x1 - x6;
    cv u25 = x2 + x5, v25 = x2 - x5;
    cv u34 = x3 + x4, v34 = x3 - x4;

    cv R1, R2, R3, I1, I2, I3;

    R1.re = fmadd(C3, u34.re, fmadd(C2, u25.re, fmadd(C1, u16.re, x0.re)));
    R1.im = fmadd(C3, u34.im, fmadd(C2, u25.im, fmadd(C1, u16.im, x0.im)));
    R2.re = fmadd(C1, u34.re, fmadd(C3, u25.re, fmadd(C2, u16.re, x0.re)));
    R2.im = fmadd(C1, u34.im, fmadd(C3, u25.im, fmadd(C2, u16.im, x0.im)));
    R3.re = fmadd(C2, u34.re, fmadd(C1, u25.re, fmadd(C3, u16.re, x0.re)));
    R3.im = fmadd(C2, u34.im, fmadd(C1, u25.im, fmadd(C3, u16.im, x0.im)));

    I1.re = fmadd(S3, v34.re, fmadd(S2, v25.re, mul(S1, v16.re)));
    I1.im = fmadd(S3, v34.im, fmadd(S2, v25.im, mul(S1, v16.im)));
    I2.re = fmsub(S2, v16.re, fmadd(S1, v34.re, mul(S3, v25.re)));
    I2.im = fmsub(S2, v16.im, fmadd(S1, v34.im, mul(S3, v25.im)));
    I3.re = fmadd(S3, v16.re, fmsub(S2, v34.re, mul(S1, v25.re)));
    I3.im = fmadd(S3, v16.im, fmsub(S2, v34.im, mul(S1, v25.im)));

    y0 = { add(add(x0.re, u16.re), add(u25.re, u34.re)),
           add(add(x0.im, u16.im), add(u25.im, u34.im)) };

    // Inverse: swap ±i on every pair.
    y1 = { sub(R1.re, I1.im), add(R1.im, I1.re) };
    y6 = { add(R1.re, I1.im), sub(R1.im, I1.re) };
    y2 = { sub(R2.re, I2.im), add(R2.im, I2.re) };
    y5 = { add(R2.re, I2.im), sub(R2.im, I2.re) };
    y3 = { sub(R3.re, I3.im), add(R3.im, I3.re) };
    y4 = { add(R3.re, I3.im), sub(R3.im, I3.re) };
}

// -- Per-M dispatchers for the butterfly + lane-shuffle steps ----------------

template <std::uint32_t M>
FFT_INLINE void pfa_butterfly_fwd(const cv* x, cv* y);

template <> FFT_INLINE void pfa_butterfly_fwd<3>(const cv* x, cv* y) {
    pfa3_butterfly_fwd(x[0], x[1], x[2], y[0], y[1], y[2]);
}
template <> FFT_INLINE void pfa_butterfly_fwd<5>(const cv* x, cv* y) {
    pfa5_butterfly_fwd(x[0], x[1], x[2], x[3], x[4],
                       y[0], y[1], y[2], y[3], y[4]);
}
template <> FFT_INLINE void pfa_butterfly_fwd<7>(const cv* x, cv* y) {
    pfa7_butterfly_fwd(x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6]);
}

template <std::uint32_t M>
FFT_INLINE void pfa_butterfly_inv(const cv* x, cv* y);

template <> FFT_INLINE void pfa_butterfly_inv<3>(const cv* x, cv* y) {
    pfa3_butterfly_inv(x[0], x[1], x[2], y[0], y[1], y[2]);
}
template <> FFT_INLINE void pfa_butterfly_inv<5>(const cv* x, cv* y) {
    pfa5_butterfly_inv(x[0], x[1], x[2], x[3], x[4],
                       y[0], y[1], y[2], y[3], y[4]);
}
template <> FFT_INLINE void pfa_butterfly_inv<7>(const cv* x, cv* y) {
    pfa7_butterfly_inv(x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6]);
}

// Forward lane shuffle: x_A lane l draws s_{(A − l) mod M} (aligned invariant).
template <std::uint32_t M>
FFT_INLINE void pfa_shuffle_fwd(cv* x, const cv* s);

template <> FFT_INLINE void pfa_shuffle_fwd<3>(cv* x, const cv* s) {
    x[0] = blend4(s[0], s[2], s[1], s[0]);
    x[1] = blend4(s[1], s[0], s[2], s[1]);
    x[2] = blend4(s[2], s[1], s[0], s[2]);
}
template <> FFT_INLINE void pfa_shuffle_fwd<5>(cv* x, const cv* s) {
    x[0] = blend4(s[0], s[4], s[3], s[2]);
    x[1] = blend4(s[1], s[0], s[4], s[3]);
    x[2] = blend4(s[2], s[1], s[0], s[4]);
    x[3] = blend4(s[3], s[2], s[1], s[0]);
    x[4] = blend4(s[4], s[3], s[2], s[1]);
}
template <> FFT_INLINE void pfa_shuffle_fwd<7>(cv* x, const cv* s) {
    x[0] = blend4(s[0], s[6], s[5], s[4]);
    x[1] = blend4(s[1], s[0], s[6], s[5]);
    x[2] = blend4(s[2], s[1], s[0], s[6]);
    x[3] = blend4(s[3], s[2], s[1], s[0]);
    x[4] = blend4(s[4], s[3], s[2], s[1]);
    x[5] = blend4(s[5], s[4], s[3], s[2]);
    x[6] = blend4(s[6], s[5], s[4], s[3]);
}

// Inverse lane shuffle (output-side): phi-slot p lane l draws y_{(p + l) mod M}.
template <std::uint32_t M>
FFT_INLINE void pfa_shuffle_inv(cv* out, const cv* y);

template <> FFT_INLINE void pfa_shuffle_inv<3>(cv* out, const cv* y) {
    out[0] = blend4(y[0], y[1], y[2], y[0]);
    out[1] = blend4(y[1], y[2], y[0], y[1]);
    out[2] = blend4(y[2], y[0], y[1], y[2]);
}
template <> FFT_INLINE void pfa_shuffle_inv<5>(cv* out, const cv* y) {
    out[0] = blend4(y[0], y[1], y[2], y[3]);
    out[1] = blend4(y[1], y[2], y[3], y[4]);
    out[2] = blend4(y[2], y[3], y[4], y[0]);
    out[3] = blend4(y[3], y[4], y[0], y[1]);
    out[4] = blend4(y[4], y[0], y[1], y[2]);
}
template <> FFT_INLINE void pfa_shuffle_inv<7>(cv* out, const cv* y) {
    out[0] = blend4(y[0], y[1], y[2], y[3]);
    out[1] = blend4(y[1], y[2], y[3], y[4]);
    out[2] = blend4(y[2], y[3], y[4], y[5]);
    out[3] = blend4(y[3], y[4], y[5], y[6]);
    out[4] = blend4(y[4], y[5], y[6], y[0]);
    out[5] = blend4(y[5], y[6], y[0], y[1]);
    out[6] = blend4(y[6], y[0], y[1], y[2]);
}

// Rotation index for the offset state after a +4 advance in lane-0 a (fwd) or
// +4 advance in natural-tile phi (inv, since phi = 4T mod M bumps by 4 per T):
//   new_q[j] = old_q[(j − 4) mod M]  →  vpermd index = (j + shift_back) mod M,
//   shift_back = (−4) mod M = (M=3: 2, M=5: 1, M=7: 3).
// Tail slots (M..7) are identity (unused).
template <std::uint32_t M> FFT_INLINE __m256i rot_idx_v();
template <> FFT_INLINE __m256i rot_idx_v<3>() {
    return _mm256_setr_epi32(2, 0, 1, 3, 4, 5, 6, 7);
}
template <> FFT_INLINE __m256i rot_idx_v<5>() {
    return _mm256_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7);
}
template <> FFT_INLINE __m256i rot_idx_v<7>() {
    return _mm256_setr_epi32(3, 4, 5, 6, 0, 1, 2, 7);
}

// Forward PFA unpack, parameterized on M ∈ {3, 5, 7}. Reads natural-order u16
// digits, writes M branches of natural-b-order AoSoV tiles into `data`.
//
// Per-iteration phase is absorbed into rotating stripe offsets held in a
// single __m256i of 8×u32 slots: q[i] always points to the stripe whose
// lane-0 has a-residue = i at the current k. Per iter: advance (vpaddd +4)
// then cyclic-relabel (vpermd via rot_idx_v<M>) to restore the invariant.
// The shuffle then uses fixed blend4 patterns with no run-time branching.
template <std::uint32_t M>
inline void fwd_pfa_unpack(double* data, std::uint32_t n,
                           const std::uint64_t* src, std::size_t limb_count)
{
    static_assert(M == 3 || M == 5 || M == 7, "PFA M must be 3, 5, or 7");
    assert((n & 3u) == 0);

    // Initial layout: at k=0, stripe m (offset m·n) has lane-0 a = m·(n mod M)
    // mod M. We want q[a] = m·n; iterate m=0..M−1 and scatter.
    alignas(32) std::uint32_t q_init[8] = {};
    std::uint32_t step = n % M;
    std::uint32_t a = 0;
    for (std::uint32_t m = 0; m < M; ++m) {
        q_init[a] = m * n;
        a += step;
        if (a >= M) a -= M;
    }
    __m256i v         = _mm256_load_si256(reinterpret_cast<const __m256i*>(q_init));
    const __m256i rot = rot_idx_v<M>();
    const __m256i bump = _mm256_set1_epi32(4);

    double* p[M];
    for (std::uint32_t i = 0; i < M; ++i)
        p[i] = data + 2u * std::size_t(n) * i;

    for (std::uint32_t t = 0; t < (n >> 2); ++t) {
        alignas(32) std::uint32_t q[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(q), v);

        cv s[M];
        for (std::uint32_t i = 0; i < M; ++i)
            s[i] = load_u16_pair(src, limb_count, q[i]);

        cv x[M];
        pfa_shuffle_fwd<M>(x, s);

        cv y[M];
        pfa_butterfly_fwd<M>(x, y);

        for (std::uint32_t i = 0; i < M; ++i) {
            store(p[i], y[i]);
            p[i] += 8;
        }

        v = _mm256_add_epi32(v, bump);
        v = _mm256_permutevar8x32_epi32(v, rot);
    }
}

// Inverse PFA butterfly with in-place natural-tile reinterpretation, for M ∈
// {3, 5, 7}. Reads M branch tiles at flat addresses {0, 2n, ..., 2n(M−1)} +
// 8t, applies the inverse radix-M butterfly along the a'-dim, shuffles
// lane-wise into M natural tiles at the SAME M flat addresses, stores in place.
//
// Output-side phi rotation is absorbed into an __m256i of 8×u32 address
// offsets (in doubles): P[p] always points to the slot whose natural-tile
// has phi = 4·T mod M equal to p. Per iter: bump all offsets by 8 (one tile
// step in doubles) and vpermd with rot_idx_v<M> — same rotation as the
// forward, since phi bumps by 4 mod M per T-step (4 is the lanes/tile).
template <std::uint32_t M>
inline void inv_pfa_butterfly(double* data, std::uint32_t n) {
    static_assert(M == 3 || M == 5 || M == 7, "PFA M must be 3, 5, or 7");
    std::uint32_t n4 = n >> 2;

    // At iter 0, branch b's tile (address 2n·b doubles) has natural-T = n4·b,
    // phi = 4·T mod M = (4·n4·b) mod M = (n·b) mod M. Scatter P[phi] = 2n·b.
    std::uint32_t phi_step = n % M;
    alignas(32) std::uint32_t P_init[8] = {};
    std::uint32_t phi = 0;
    for (std::uint32_t b = 0; b < M; ++b) {
        P_init[phi] = 2u * n * b;
        phi += phi_step;
        if (phi >= M) phi -= M;
    }
    __m256i V         = _mm256_load_si256(reinterpret_cast<const __m256i*>(P_init));
    const __m256i rot = rot_idx_v<M>();
    const __m256i bump = _mm256_set1_epi32(8);

    // Branch-tile loads walk monotonically at (branches 0..M-1, tile t).
    double* p[M];
    for (std::uint32_t i = 0; i < M; ++i)
        p[i] = data + 2u * std::size_t(n) * i;

    for (std::uint32_t t = 0; t < n4; ++t) {
        cv x[M];
        for (std::uint32_t i = 0; i < M; ++i) {
            x[i] = load_cv(p[i]);
            p[i] += 8;
        }

        cv y[M];
        pfa_butterfly_inv<M>(x, y);

        cv out[M];
        pfa_shuffle_inv<M>(out, y);

        alignas(32) std::uint32_t P[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(P), V);
        for (std::uint32_t i = 0; i < M; ++i)
            store(data + P[i], out[i]);

        V = _mm256_add_epi32(V, bump);
        V = _mm256_permutevar8x32_epi32(V, rot);
    }
}

// -----------------------------------------------------------------------------
// Forward radix-2^2 cascade (cache-blocked). Runs DIF passes from length
// `len_start` down to the tail. `n` is the per-branch FFT size; data points
// to that branch's buffer.
// -----------------------------------------------------------------------------

inline void fwd_cascade(double* data, std::uint32_t n, const plan& pl,
                        std::uint32_t len_start, unsigned lg_len_start)
{
    unsigned lgn = ctz_u32(n);
    std::uint32_t blk = 256u << (lgn & 1u);
    std::uint32_t tail_blk = 16u >> (lgn & 1u);
    const double* tw16 = stage_tw_if_present(pl, 4);

    if (blk > n) blk = n;

    std::uint32_t len = len_start;
    unsigned lg_len = lg_len_start;
    for (; len > blk; len >>= 2, lg_len -= 2u)
        fwd_range(data, 0, n, len, stage_tw(pl, lg_len));

    for (std::uint32_t base = 0; base < n; base += blk) {
        std::uint32_t cur = len;
        unsigned cur_lg = lg_len;
        for (; cur > tail_blk; cur >>= 2, cur_lg -= 2u)
            fwd_range(data, base, base + blk, cur, stage_tw(pl, cur_lg));
        fwd_tail_range(data, base, base + blk, tw16);
    }
}

// -----------------------------------------------------------------------------
// Inverse radix-2^2 cascade (cache-blocked). Mirror of fwd_cascade.
// -----------------------------------------------------------------------------

inline void inv_cascade(double* data, std::uint32_t n, const plan& pl) {
    unsigned lgn = ctz_u32(n);
    std::uint32_t blk = 256u << (lgn & 1u);
    std::uint32_t tail_blk = 16u >> (lgn & 1u);
    const double* tw16 = stage_tw_if_present(pl, 4);

    if (blk > n) blk = n;

    for (std::uint32_t base = 0; base < n; base += blk) {
        inv_tail_range(data, base, base + blk, tw16);
        for (std::uint32_t cur = tail_blk << 2; cur <= blk; cur <<= 2)
            inv_range(data, base, base + blk, cur, stage_tw(pl, ctz_u32(cur)));
    }

    for (std::uint32_t len = blk << 2; len <= n; len <<= 2)
        inv_range(data, 0, n, len, stage_tw(pl, ctz_u32(len)));
}

// -----------------------------------------------------------------------------
// Forward FFT (thin dispatcher). `n` is the per-branch size (== full N for
// pl.M == 1; full N / 3 for pl.M == 3).
// -----------------------------------------------------------------------------

inline void fwd(double* data, const std::uint64_t* limbs, std::size_t limb_count,
                std::uint32_t n, const plan& pl)
{
    unsigned lgn = ctz_u32(n);

    if (pl.M != 1u) {
        // PFA path: radix-M input stage writes natural-b-order per-branch tiles,
        // then a full r22 cascade per branch (from len = n down).
        switch (pl.M) {
            case 3: fwd_pfa_unpack<3>(data, n, limbs, limb_count); break;
            case 5: fwd_pfa_unpack<5>(data, n, limbs, limb_count); break;
            case 7: fwd_pfa_unpack<7>(data, n, limbs, limb_count); break;
            default: assert(!"invalid PFA M"); return;
        }
        for (std::uint32_t b = 0; b < pl.M; ++b)
            fwd_cascade(data + 2u * std::size_t(n) * b, n, pl, n, lgn);
        return;
    }

    if (n < 32u) {
        std::uint32_t tiles = (n + 3u) >> 2;
        for (std::uint32_t t = 0; t < tiles; ++t)
            store(data + 8u * t, load_u16_pair(limbs, limb_count, 4u * t));
        if (n == 2u) r2_stage(data, 2u);
        else         fwd_tail_range(data, 0, n, stage_tw_if_present(pl, 4));
        return;
    }

    unpack_fwd(data, n, limbs, limb_count, stage_tw(pl, lgn));
    fwd_cascade(data, n, pl, n >> 2, lgn - 2u);
}

// -----------------------------------------------------------------------------
// Inverse FFT (thin dispatcher).
// -----------------------------------------------------------------------------

inline void inv(double* data, std::uint32_t n, const plan& pl) {
    if (pl.M != 1u) {
        // PFA path: per-branch r22 DIT cascade, then in-place inverse radix-M
        // butterfly + natural-tile reinterpretation.
        for (std::uint32_t b = 0; b < pl.M; ++b)
            inv_cascade(data + 2u * std::size_t(n) * b, n, pl);
        switch (pl.M) {
            case 3: inv_pfa_butterfly<3>(data, n); break;
            case 5: inv_pfa_butterfly<5>(data, n); break;
            case 7: inv_pfa_butterfly<7>(data, n); break;
            default: assert(!"invalid PFA M"); return;
        }
        return;
    }

    if (n < 32u) {
        if (n == 2u) r2_stage(data, 2u);
        else         inv_tail_range(data, 0, n, stage_tw_if_present(pl, 4));
        return;
    }
    inv_cascade(data, n, pl);
}

// -----------------------------------------------------------------------------
// PQ pointwise evaluation.
//
// The right-angle convolution trick: a bigint A of N u16 digits is viewed as
// P + i*Q where P = even digits, Q = odd digits. Convolution of A*B in u16
// needs to blend P_A*P_B, Q_A*Q_B, P_A*Q_B + Q_A*P_B suitably. With the N/2
// complex transform X = FFT(A), the neighbor pair (X[k], X[N/2-k]*) gives the
// two symmetric P/Q combinations; one twiddle rotates them.
//
// Given X[k], X_n = reverse(conj(X[N/2-k])) packed into the same tile,
// and similarly Y[k], Y_n:
//   pq  = X * Y                    (the usual complex mul)
//   dp  = X - X_n^                  ... where (^) flips imag sign
//   dq  = Y - Y_n^
//   c   = (dp * dq) * w
// output = (pq - 0.25 * c) * (1/n)
//
// Scaling is a precomputed multiplicative factor (scale = 2^-lg(n)), which is
// cheaper and simpler than the old exponent-bit trick.
// -----------------------------------------------------------------------------

FFT_INLINE void pq_eval(double& zr, double& zi,
                        double xr, double xi, double xnr, double xni,
                        double yr, double yi, double ynr, double yni,
                        double wr, double wi, double scale, double qscale)
{
    double pqr = xr * yr - xi * yi;
    double pqi = xr * yi + xi * yr;
    double dpr = xr - xnr, dpi = xi + xni;
    double dqr = yr - ynr, dqi = yi + yni;
    double tr  = dpr * dqr - dpi * dqi;
    double ti  = dpr * dqi + dpi * dqr;
    double cr  = tr * wr - ti * wi;
    double ci  = tr * wi + ti * wr;
    zr = pqr * scale - qscale * cr;
    zi = pqi * scale - qscale * ci;
}

// Paired pointwise eval: computes both "left" and "right" outputs while
// sharing the cross-term (dp*dq)*w. Substituting (xn,x,yn,y,conj(w)) into
// the left formula yields dp'=(-dpr, dpi), t'=(tr,-ti), c'=(cr,-ci); only
// the primary product x*y differs between the two calls.
FFT_INLINE void pq_eval4_pair(cv& outl, cv& outr,
                              cv x, cv xn, cv y, cv yn, cv w,
                              vec4 scale, vec4 qscale)
{
    cv pql = cmul(x,  y);
    cv pqr = cmul(xn, yn);
    cv dp  = { sub(x.re, xn.re), add(x.im, xn.im) };
    cv dq  = { sub(y.re, yn.re), add(y.im, yn.im) };
    cv t   = cmul(dp, dq);
    cv c   = cmul(t, w);
    vec4 qcr = mul(qscale, c.re);
    vec4 qci = mul(qscale, c.im);
    outl = { fmsub(pql.re, scale, qcr), fmsub(pql.im, scale, qci) };
    // Right uses c' = (cr, -ci): re still subtracts qcr; im ADDS qci.
    outr = { fmsub(pqr.re, scale, qcr), fmadd(pqr.im, scale, qci) };
}

FFT_INLINE void pq_sqr(double& zr, double& zi,
                       double xr, double xi, double xnr, double xni,
                       double wr, double wi, double scale, double qscale)
{
    double pqr = xr * xr - xi * xi;
    double pqi = (xr + xr) * xi;
    double dpr = xr - xnr, dpi = xi + xni;
    double tr  = dpr * dpr - dpi * dpi;
    double ti  = (dpr + dpr) * dpi;
    double cr  = tr * wr - ti * wi;
    double ci  = tr * wi + ti * wr;
    zr = pqr * scale - qscale * cr;
    zi = pqi * scale - qscale * ci;
}

// Paired sqr variant: shares dp*dp and c = (dp*dp)*w between the two outputs.
// dp' = (-dpr, dpi) so (dp'^2).re = (dp^2).re, (dp'^2).im = -(dp^2).im.
// With w_conj = (wr, -wi): c'.r = cr, c'.i = -ci.
FFT_INLINE void pq_sqr4_pair(cv& outl, cv& outr,
                             cv x, cv xn, cv w,
                             vec4 scale, vec4 qscale)
{
    cv pql, pqr;
    pql.re = fmsub(x.re,  x.re,  mul(x.im,  x.im));
    pql.im = mul(add(x.re,  x.re),  x.im);
    pqr.re = fmsub(xn.re, xn.re, mul(xn.im, xn.im));
    pqr.im = mul(add(xn.re, xn.re), xn.im);
    cv dp = { sub(x.re, xn.re), add(x.im, xn.im) };
    cv t;
    t.re = fmsub(dp.re, dp.re, mul(dp.im, dp.im));
    t.im = mul(add(dp.re, dp.re), dp.im);
    cv c = cmul(t, w);
    vec4 qcr = mul(qscale, c.re);
    vec4 qci = mul(qscale, c.im);
    outl = { fmsub(pql.re, scale, qcr), fmsub(pql.im, scale, qci) };
    outr = { fmsub(pqr.re, scale, qcr), fmadd(pqr.im, scale, qci) };
}

// PQ twiddle: the rotation factor for the (k, N/2-k) partner pair.
FFT_INLINE cv pq_twiddle4(const plan& pl, std::uint32_t block4_index) {
    const double* pq_re = pl.pq_omega_br;
    const double* pq_im = pl.pq_omega_br + (std::size_t(pl.pq_omega_br_n) >> 2);
    double gr = pq_re[block4_index];
    double gi = pq_im[block4_index];
    return { set_r4(1.0 + gr, 1.0 - gr, 1.0 - gi, 1.0 + gi),
             set_r4(gi,       -gi,      gr,       -gr      ) };
}

FFT_INLINE void pq_twiddle(double& wr, double& wi, const plan& pl, std::uint32_t i) {
    const double* pq_re = pl.pq_omega_br;
    const double* pq_im = pl.pq_omega_br + (std::size_t(pl.pq_omega_br_n) >> 2);
    double gr = pq_re[i >> 2];
    double gi = pq_im[i >> 2];
    switch (i & 3u) {
      case 0: wr = 1.0 + gr; wi =  gi; break;
      case 1: wr = 1.0 - gr; wi = -gi; break;
      case 2: wr = 1.0 - gi; wi =  gr; break;
      default: wr = 1.0 + gi; wi = -gr; break;
    }
}

// Branch-scoped variants: multiply g_base by (w3r, w3i) = ω_3^{k2} before
// building the 1 ± g / 1 ± i·g lane pattern. Used for PFA cross-pair PQ.
FFT_INLINE cv pq_twiddle4_branch(const plan& pl, std::uint32_t block4_index,
                                 double w3r, double w3i)
{
    const double* pq_re = pl.pq_omega_br;
    const double* pq_im = pl.pq_omega_br + (std::size_t(pl.pq_omega_br_n) >> 2);
    double g0r = pq_re[block4_index];
    double g0i = pq_im[block4_index];
    double gr  = g0r * w3r - g0i * w3i;
    double gi  = g0r * w3i + g0i * w3r;
    return { set_r4(1.0 + gr, 1.0 - gr, 1.0 - gi, 1.0 + gi),
             set_r4(gi,       -gi,      gr,       -gr      ) };
}

FFT_INLINE void pq_twiddle_branch(double& wr, double& wi, const plan& pl,
                                  std::uint32_t i, double w3r, double w3i)
{
    const double* pq_re = pl.pq_omega_br;
    const double* pq_im = pl.pq_omega_br + (std::size_t(pl.pq_omega_br_n) >> 2);
    double g0r = pq_re[i >> 2];
    double g0i = pq_im[i >> 2];
    double gr  = g0r * w3r - g0i * w3i;
    double gi  = g0r * w3i + g0i * w3r;
    switch (i & 3u) {
      case 0: wr = 1.0 + gr; wi =  gi; break;
      case 1: wr = 1.0 - gr; wi = -gi; break;
      case 2: wr = 1.0 - gi; wi =  gr; break;
      default: wr = 1.0 + gi; wi = -gr; break;
    }
}

// -----------------------------------------------------------------------------
// Pointwise PQ multiply in bit-reversed order. Both inputs must already be
// forward-FFT'd; result replaces `data`.
//
// Handles the first 8 points scalar (each point's partner is within this 8),
// then vectorizes 4 lanes at a time for the rest, pairing [k, n-1-(k&-4)].
// -----------------------------------------------------------------------------

// Self-pair PQ multiply scan: pairs (i, partner[i]) within one branch buffer.
// scale_d = 1 / N_full (N_full = n for pow2, M·n for PFA b0 self-pair).
inline void pq_mul_self_scan(double* data, double* other, std::uint32_t n,
                             const plan& pl, double scale_d)
{
    double qscale_d = 0.25 * scale_d;
    vec4   scale    = splat(scale_d);
    vec4   qscale   = splat(qscale_d);

    // Head = positions 0..7. Within-branch partner map is {0,1,3,2,7,6,5,4}:
    // tile 0 partner permutation = swap23 (lanes 2,3); tile 1 = reverse.
    // Pq_eval4_pair's "partner_w = conj(primary_w)" assumption holds here by
    // the bit-reversal construction of pq_omega_br.
    if (n >= 8u) {
        cv x0 = load_cv(data  + 0);
        cv x1 = load_cv(data  + 8);
        cv y0 = load_cv(other + 0);
        cv y1 = load_cv(other + 8);

        cv w0 = pq_twiddle4(pl, 0);
        cv w1 = pq_twiddle4(pl, 1);

        cv outl0, outr0, outl1, outr1;
        pq_eval4_pair(outl0, outr0, x0, swap23(x0),  y0, swap23(y0),  w0, scale, qscale);
        pq_eval4_pair(outl1, outr1, x1, reverse(x1), y1, reverse(y1), w1, scale, qscale);

        // outr duplicates outl under the partner permutation (self-pair); only
        // store outl, which covers both halves of each partnering pair.
        store(data + 0, outl0);
        store(data + 8, outl1);
    } else {
        static constexpr unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t j = partner[i];
            if (i > j) continue;

            double xr  = re_at(data, i),  xi  = im_at(data, i);
            double xjr = re_at(data, j),  xji = im_at(data, j);
            double yr  = re_at(other, i), yi  = im_at(other, i);
            double yjr = re_at(other, j), yji = im_at(other, j);

            double wr, wi, zr, zi;
            pq_twiddle(wr, wi, pl, i);
            pq_eval(zr, zi, xr, xi, xjr, xji, yr, yi, yjr, yji, wr, wi, scale_d, qscale_d);
            set_at(data, i, zr, zi);

            if (i != j) {
                pq_twiddle(wr, wi, pl, j);
                pq_eval(zr, zi, xjr, xji, xr, xi, yjr, yji, yr, yi, wr, wi, scale_d, qscale_d);
                set_at(data, j, zr, zi);
            }
        }
    }

    for (std::uint32_t base = 8u; base < n; base <<= 1) {
        std::uint32_t grp = base;
        for (std::uint32_t i = 0; i < (grp >> 1); i += 4u) {
            std::uint32_t li = base + i;
            std::uint32_t ri = (base + grp) - 4u - i;

            cv xl = load_cv(tile_at(data,  li));
            cv xr2= load_cv(tile_at(data,  ri));
            cv yl = load_cv(tile_at(other, li));
            cv yr2= load_cv(tile_at(other, ri));

            cv xn = reverse(xr2);
            cv yn = reverse(yr2);

            cv w = pq_twiddle4(pl, li >> 2);

            cv outl, outr;
            pq_eval4_pair(outl, outr, xl, xn, yl, yn, w, scale, qscale);

            store(tile_at(data, li), outl);
            store(tile_at(data, ri), reverse(outr));
        }
    }
}

// Cross-pair PQ multiply scan: primary at (b1, i), partner at (b2, partner[i]).
// Twiddle is g_base * ω_M^k; partner uses conj(w) via pq_eval4_pair symmetry.
inline void pq_mul_cross_scan(double* d_left, double* d_right,
                              double* o_left, double* o_right,
                              std::uint32_t n, const plan& pl,
                              double scale_d, double w3r, double w3i)
{
    double qscale_d = 0.25 * scale_d;
    vec4   scale    = splat(scale_d);
    vec4   qscale   = splat(qscale_d);

    if (n >= 8u) {
        // Head vectorized: primary in natural order from d_left/o_left;
        // partner loaded from d_right/o_right with the partner permutation
        // (tile 0 = swap23, tile 1 = reverse). Partner store uses the same
        // (self-inverse) permutations.
        cv xl0 = load_cv(d_left  + 0);
        cv xl1 = load_cv(d_left  + 8);
        cv yl0 = load_cv(o_left  + 0);
        cv yl1 = load_cv(o_left  + 8);
        cv xr0 = swap23 (load_cv(d_right + 0));
        cv xr1 = reverse(load_cv(d_right + 8));
        cv yr0 = swap23 (load_cv(o_right + 0));
        cv yr1 = reverse(load_cv(o_right + 8));

        cv w0 = pq_twiddle4_branch(pl, 0, w3r, w3i);
        cv w1 = pq_twiddle4_branch(pl, 1, w3r, w3i);

        cv outl0, outr0, outl1, outr1;
        pq_eval4_pair(outl0, outr0, xl0, xr0, yl0, yr0, w0, scale, qscale);
        pq_eval4_pair(outl1, outr1, xl1, xr1, yl1, yr1, w1, scale, qscale);

        store(d_left  + 0, outl0);
        store(d_left  + 8, outl1);
        store(d_right + 0, swap23(outr0));
        store(d_right + 8, reverse(outr1));
    } else {
        static constexpr unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
        double w3r_part = w3r;
        double w3i_part = -w3i;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t j = partner[i];

            double xr  = re_at(d_left,  i), xi  = im_at(d_left,  i);
            double xjr = re_at(d_right, j), xji = im_at(d_right, j);
            double yr  = re_at(o_left,  i), yi  = im_at(o_left,  i);
            double yjr = re_at(o_right, j), yji = im_at(o_right, j);

            double wr, wi, zr, zi;
            pq_twiddle_branch(wr, wi, pl, i, w3r, w3i);
            pq_eval(zr, zi, xr, xi, xjr, xji, yr, yi, yjr, yji, wr, wi, scale_d, qscale_d);
            set_at(d_left, i, zr, zi);

            pq_twiddle_branch(wr, wi, pl, j, w3r_part, w3i_part);
            pq_eval(zr, zi, xjr, xji, xr, xi, yjr, yji, yr, yi, wr, wi, scale_d, qscale_d);
            set_at(d_right, j, zr, zi);
        }
    }

    // Cross-pair iterates the FULL grp range (not grp/2): unlike self-pair
    // where (li, ri) both live in one buffer and one iteration covers both,
    // cross-pair's primary (b1) and partner (b2) are in different buffers,
    // so pairs (b1[li], b2[ri]) with li < ri AND (b1[li], b2[ri]) with li > ri
    // are all distinct and must all be processed.
    for (std::uint32_t base = 8u; base < n; base <<= 1) {
        std::uint32_t grp = base;
        for (std::uint32_t i = 0; i < grp; i += 4u) {
            std::uint32_t li = base + i;
            std::uint32_t ri = (base + grp) - 4u - i;

            cv xl = load_cv(tile_at(d_left,  li));
            cv xr2= load_cv(tile_at(d_right, ri));
            cv yl = load_cv(tile_at(o_left,  li));
            cv yr2= load_cv(tile_at(o_right, ri));

            cv xn = reverse(xr2);
            cv yn = reverse(yr2);

            cv w = pq_twiddle4_branch(pl, li >> 2, w3r, w3i);

            cv outl, outr;
            pq_eval4_pair(outl, outr, xl, xn, yl, yn, w, scale, qscale);

            store(tile_at(d_left,  li), outl);
            store(tile_at(d_right, ri), reverse(outr));
        }
    }
}

// Antipodal-pair table: branch b's partner is (M − b) mod M (for the k2 pair
// (k2, M−k2) whose PQ outputs are conjugate-related). Pairs are (1, M−1),
// (2, M−2), etc.; b=0 always self-pairs.
inline void pointwise_mul(double* data, double* other,
                          std::uint32_t n, const plan& pl)
{
    double scale_d = std::ldexp(1.0, -int(ctz_u32(n)));
    if (pl.M == 1) {
        pq_mul_self_scan(data, other, n, pl, scale_d);
        return;
    }
    // PFA radix-M: full-N = M·n, so scale = 1/(M·n).
    scale_d *= (1.0 / double(pl.M));
    // Branch 0 (k2 = 0, W^0 = 1): self-pair.
    pq_mul_self_scan(data, other, n, pl, scale_d);
    // Cross-pairs (b, M−b) for b = 1..(M−1)/2, each with ω_M^b twiddle.
    const double* wr_tab = (pl.M == 3) ? W3_RE : (pl.M == 5) ? W5_RE : W7_RE;
    const double* wi_tab = (pl.M == 3) ? W3_IM : (pl.M == 5) ? W5_IM : W7_IM;
    for (std::uint32_t b = 1; b <= pl.M / 2u; ++b) {
        std::uint32_t bp = pl.M - b;
        pq_mul_cross_scan(data  + 2u*std::size_t(n)*b,  data  + 2u*std::size_t(n)*bp,
                          other + 2u*std::size_t(n)*b,  other + 2u*std::size_t(n)*bp,
                          n, pl, scale_d, wr_tab[b], wi_tab[b]);
    }
}

// Self-pair PQ square scan (analog of pq_mul_self_scan).
inline void pq_sqr_self_scan(double* data, std::uint32_t n, const plan& pl,
                             double scale_d)
{
    double qscale_d = 0.25 * scale_d;
    vec4   scale    = splat(scale_d);
    vec4   qscale   = splat(qscale_d);

    if (n >= 8u) {
        cv x0 = load_cv(data + 0);
        cv x1 = load_cv(data + 8);
        cv w0 = pq_twiddle4(pl, 0);
        cv w1 = pq_twiddle4(pl, 1);
        cv outl0, outr0, outl1, outr1;
        pq_sqr4_pair(outl0, outr0, x0, swap23(x0),  w0, scale, qscale);
        pq_sqr4_pair(outl1, outr1, x1, reverse(x1), w1, scale, qscale);
        store(data + 0, outl0);
        store(data + 8, outl1);
    } else {
        static constexpr unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t j = partner[i];
            if (i > j) continue;

            double xr  = re_at(data, i),  xi  = im_at(data, i);
            double xjr = re_at(data, j),  xji = im_at(data, j);

            double wr, wi, zr, zi;
            pq_twiddle(wr, wi, pl, i);
            pq_sqr(zr, zi, xr, xi, xjr, xji, wr, wi, scale_d, qscale_d);
            set_at(data, i, zr, zi);

            if (i != j) {
                pq_twiddle(wr, wi, pl, j);
                pq_sqr(zr, zi, xjr, xji, xr, xi, wr, wi, scale_d, qscale_d);
                set_at(data, j, zr, zi);
            }
        }
    }

    for (std::uint32_t base = 8u; base < n; base <<= 1) {
        std::uint32_t grp = base;
        for (std::uint32_t i = 0; i < (grp >> 1); i += 4u) {
            std::uint32_t li = base + i;
            std::uint32_t ri = (base + grp) - 4u - i;

            cv xl = load_cv(tile_at(data, li));
            cv xr2= load_cv(tile_at(data, ri));
            cv xn = reverse(xr2);

            cv w = pq_twiddle4(pl, li >> 2);

            cv outl, outr;
            pq_sqr4_pair(outl, outr, xl, xn, w, scale, qscale);

            store(tile_at(data, li), outl);
            store(tile_at(data, ri), reverse(outr));
        }
    }
}

// Cross-pair PQ square scan.
inline void pq_sqr_cross_scan(double* d_left, double* d_right,
                              std::uint32_t n, const plan& pl,
                              double scale_d, double w3r, double w3i)
{
    double qscale_d = 0.25 * scale_d;
    vec4   scale    = splat(scale_d);
    vec4   qscale   = splat(qscale_d);

    if (n >= 8u) {
        cv xl0 = load_cv(d_left  + 0);
        cv xl1 = load_cv(d_left  + 8);
        cv xr0 = swap23 (load_cv(d_right + 0));
        cv xr1 = reverse(load_cv(d_right + 8));

        cv w0 = pq_twiddle4_branch(pl, 0, w3r, w3i);
        cv w1 = pq_twiddle4_branch(pl, 1, w3r, w3i);

        cv outl0, outr0, outl1, outr1;
        pq_sqr4_pair(outl0, outr0, xl0, xr0, w0, scale, qscale);
        pq_sqr4_pair(outl1, outr1, xl1, xr1, w1, scale, qscale);

        store(d_left  + 0, outl0);
        store(d_left  + 8, outl1);
        store(d_right + 0, swap23(outr0));
        store(d_right + 8, reverse(outr1));
    } else {
        static constexpr unsigned char partner[8] = { 0, 1, 3, 2, 7, 6, 5, 4 };
        double w3r_part = w3r;
        double w3i_part = -w3i;
        for (std::uint32_t i = 0; i < n; ++i) {
            std::uint32_t j = partner[i];

            double xr  = re_at(d_left,  i), xi  = im_at(d_left,  i);
            double xjr = re_at(d_right, j), xji = im_at(d_right, j);

            double wr, wi, zr, zi;
            pq_twiddle_branch(wr, wi, pl, i, w3r, w3i);
            pq_sqr(zr, zi, xr, xi, xjr, xji, wr, wi, scale_d, qscale_d);
            set_at(d_left, i, zr, zi);

            pq_twiddle_branch(wr, wi, pl, j, w3r_part, w3i_part);
            pq_sqr(zr, zi, xjr, xji, xr, xi, wr, wi, scale_d, qscale_d);
            set_at(d_right, j, zr, zi);
        }
    }

    // See note on pq_mul_cross_scan's full-range iteration.
    for (std::uint32_t base = 8u; base < n; base <<= 1) {
        std::uint32_t grp = base;
        for (std::uint32_t i = 0; i < grp; i += 4u) {
            std::uint32_t li = base + i;
            std::uint32_t ri = (base + grp) - 4u - i;

            cv xl = load_cv(tile_at(d_left,  li));
            cv xr2= load_cv(tile_at(d_right, ri));
            cv xn = reverse(xr2);

            cv w = pq_twiddle4_branch(pl, li >> 2, w3r, w3i);

            cv outl, outr;
            pq_sqr4_pair(outl, outr, xl, xn, w, scale, qscale);

            store(tile_at(d_left,  li), outl);
            store(tile_at(d_right, ri), reverse(outr));
        }
    }
}

inline void pointwise_sqr(double* data, std::uint32_t n, const plan& pl) {
    double scale_d = std::ldexp(1.0, -int(ctz_u32(n)));
    if (pl.M == 1) {
        pq_sqr_self_scan(data, n, pl, scale_d);
        return;
    }
    scale_d *= (1.0 / double(pl.M));
    pq_sqr_self_scan(data, n, pl, scale_d);  // b0 self-pair
    const double* wr_tab = (pl.M == 3) ? W3_RE : (pl.M == 5) ? W5_RE : W7_RE;
    const double* wi_tab = (pl.M == 3) ? W3_IM : (pl.M == 5) ? W5_IM : W7_IM;
    for (std::uint32_t b = 1; b <= pl.M / 2u; ++b) {
        std::uint32_t bp = pl.M - b;
        pq_sqr_cross_scan(data + 2u*std::size_t(n)*b, data + 2u*std::size_t(n)*bp,
                          n, pl, scale_d, wr_tab[b], wi_tab[b]);
    }
}

// -----------------------------------------------------------------------------
// Limb recovery from PQ FFT output. The inverse FFT leaves each tile as 4
// pairs of (even_digit, odd_digit) in double form; we clip negatives, round
// via the 2^52 magic, merge even + (odd << 16) in 64 bits, then group two
// pairs into one __int128 partial (low 64 = sum of 32-bit halves, high 64 =
// carry) and propagate the carry chain across limbs.
// -----------------------------------------------------------------------------

FFT_INLINE void u64_to_i128x2(unsigned __int128 part[2], __m256i u_i) {
    const __m256i even_mask = _mm256_setr_epi64x(-1, 0, -1, 0);
    const __m256i one_mask  = _mm256_setr_epi64x( 1, 0,  1, 0);
    const __m256i sign_mask = _mm256_set1_epi64x(0x8000000000000000ULL);

    __m256i even_u  = _mm256_and_si256(u_i, even_mask);
    __m256i odd_dup = _mm256_permute4x64_epi64(u_i, 0xF5);
    __m256i odd_lo  = _mm256_and_si256(_mm256_slli_epi64(odd_dup, 32), even_mask);
    __m256i odd_hi  = _mm256_and_si256(_mm256_srli_epi64(odd_dup, 32), even_mask);
    __m256i sum_lo  = _mm256_add_epi64(even_u, odd_lo);
    __m256i carry_cmp = _mm256_cmpgt_epi64(_mm256_xor_si256(even_u,  sign_mask),
                                           _mm256_xor_si256(sum_lo, sign_mask));
    __m256i carry64 = _mm256_and_si256(carry_cmp, one_mask);
    __m256i sum_hi  = _mm256_add_epi64(odd_hi, carry64);

    alignas(32) std::uint64_t lo_tmp[4];
    alignas(32) std::uint64_t hi_tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lo_tmp), sum_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(hi_tmp), sum_hi);
    part[0] = (unsigned __int128)lo_tmp[0] + ((unsigned __int128)hi_tmp[0] << 64);
    part[1] = (unsigned __int128)lo_tmp[2] + ((unsigned __int128)hi_tmp[2] << 64);
}

inline int recover_limbs(std::uint64_t* rp, const double* data,
                         std::ptrdiff_t an, std::ptrdiff_t bn)
{
    const vec4 zero = zero4();
    const __m256i bias_bits = _mm256_set1_epi64x(0x4330000000000000ULL);
    const vec4    bias52    = _mm256_castsi256_pd(bias_bits);
    std::size_t result_limbs = std::size_t(an + bn);
    std::size_t total_digits = 4u * std::size_t(an + bn);
    unsigned __int128 carry = 0;

    std::memset(rp, 0, result_limbs * sizeof(std::uint64_t));

    for (std::size_t d = 0; d < total_digits; d += 8u) {
        const double* tile = tile_at(data, std::uint32_t(d >> 1));
        vec4 re = load(tile + 0);
        vec4 im = load(tile + 4);

        std::size_t rem = total_digits - d;
        if (rem > 8u) rem = 8u;

        re = _mm256_max_pd(re, zero);
        im = _mm256_max_pd(im, zero);
        __m256i re_i = _mm256_sub_epi64(_mm256_castpd_si256(add(re, bias52)), bias_bits);
        __m256i im_i = _mm256_sub_epi64(_mm256_castpd_si256(add(im, bias52)), bias_bits);
        __m256i u_i  = _mm256_add_epi64(re_i, _mm256_slli_epi64(im_i, 16));

        std::size_t limb_idx = d >> 2;
        if (rem >= 8u) {
            unsigned __int128 part[2];
            u64_to_i128x2(part, u_i);
            carry += part[0];
            rp[limb_idx] = std::uint64_t(carry); carry >>= 64; ++limb_idx;
            carry += part[1];
            rp[limb_idx] = std::uint64_t(carry); carry >>= 64;
        } else if (rem >= 4u) {
            alignas(32) std::uint64_t u64[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(u64), u_i);
            unsigned __int128 p0 = (unsigned __int128)u64[0]
                                 + ((unsigned __int128)u64[1] << 32);
            carry += p0;
            rp[limb_idx] = std::uint64_t(carry); carry >>= 64;
        }
    }

    return carry == 0;
}

// -----------------------------------------------------------------------------
// Thread-local singletons.
// -----------------------------------------------------------------------------

thread_local workspace ws;
thread_local plan      pl;

}  // anonymous namespace

// -----------------------------------------------------------------------------
// Public API.
// -----------------------------------------------------------------------------

// Choose the smallest supported FFT size (returns {N_full, n_branch, M}).
// Candidates: pow2 N, or PFA N = M·2^L for M ∈ {3, 5, 7}. Pick strictly
// smallest N_full; ties prefer smaller M (simpler path).
//
// Threshold: PFA-7's direct-form butterfly is ~5–7% slower per-limb than
// pow-2 / PFA-3/5 (18 cv·real mults on Zen4 AVX2+FMA). Below n_branch=2048
// the 12.5% size-grid savings don't offset the butterfly cost — empirical
// data show e.g. 1792 limbs @ PFA-7-7168 ≈ 2048 limbs @ pow2-8192 in total
// wall-clock, so picking pow2-8192 (simpler path, cache-friendlier) is a
// better choice at that size. Gate PFA-7 until n_branch ≥ 2048.
namespace {
struct fft_size { std::uint32_t N_full; std::uint32_t n_branch; std::uint32_t M; };
inline fft_size choose_fft_size(std::uint32_t needed) {
    fft_size best = { ceil_pow2(needed), ceil_pow2(needed), 1u };
    for (std::uint32_t M : { 3u, 5u, 7u }) {
        std::uint32_t n_br     = ceil_pow2((needed + M - 1u) / M);
        std::uint32_t N        = M * n_br;
        std::uint32_t min_n_br = (M == 7u) ? 2048u : 4u;
        if (n_br >= min_n_br && N >= needed && N < best.N_full)
            best = { N, n_br, M };
    }
    return best;
}
}  // namespace

int fft::mul(std::uint64_t* rp,
             const std::uint64_t* ap, std::ptrdiff_t an,
             const std::uint64_t* bp, std::ptrdiff_t bn)
{
    if (an <= 0 || bn <= 0) return 0;

    std::uint32_t na = 4u * std::uint32_t(an);
    std::uint32_t nb = 4u * std::uint32_t(bn);
    fft_size fs = choose_fft_size((na + nb + 1u) >> 1);
    if (fs.n_branch > 65536u) return 0;

    ensure(pl, fs.n_branch);
    pl.M = fs.M;
    ensure(ws, fs.N_full);

    fwd(ws.data,  ap, std::size_t(an), fs.n_branch, pl);
    fwd(ws.data2, bp, std::size_t(bn), fs.n_branch, pl);

    pointwise_mul(ws.data, ws.data2, fs.n_branch, pl);
    inv(ws.data, fs.n_branch, pl);

    return recover_limbs(rp, ws.data, an, bn);
}

int fft::sqr(std::uint64_t* rp, const std::uint64_t* ap, std::ptrdiff_t an) {
    if (an <= 0) return 0;

    std::uint32_t na = 4u * std::uint32_t(an);
    fft_size fs = choose_fft_size(na);  // na+na+1 >> 1 = na rounded up
    if (fs.n_branch > 65536u) return 0;

    ensure(pl, fs.n_branch);
    pl.M = fs.M;
    ensure(ws, fs.N_full);

    fwd(ws.data, ap, std::size_t(an), fs.n_branch, pl);
    pointwise_sqr(ws.data, fs.n_branch, pl);
    inv(ws.data, fs.n_branch, pl);

    return recover_limbs(rp, ws.data, an, an);
}

#endif  // INT_FFT_IMPLEMENTATION

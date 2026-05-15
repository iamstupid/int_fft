// Correctness smoke test for the centered U16 codec band.
//
// Build:
//   clang++-21 -O3 -mavx2 -mfma -std=c++17 -Iexperimental/int_fft/src \
//       experimental/int_fft/bench/check_centered_u16.cpp \
//       experimental/int_fft/src/fft.cpp -lgmp -lm -o /tmp/check_centered_u16

#include "fft.hpp"

#include <gmp.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void fill_limbs(std::vector<std::uint64_t>& v, std::mt19937_64& rng)
{
    for (std::uint64_t& x : v) x = rng();
    if (!v.empty()) v.back() |= 1ull << 63;
}

static int first_mismatch(const std::vector<std::uint64_t>& a,
                          const std::vector<std::uint64_t>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return int(i);
    return -1;
}

int main()
{
    std::mt19937_64 rng(0x123456789abcdef0ull);
    const int sizes[] = {
        16384, 16385, 17000, 20475, 21845, 24576, 25594, 30000, 32768,
        32769, 45000, 65536, 131072
    };

    for (int n : sizes) {
        std::vector<std::uint64_t> a(n), b(n);
        std::vector<std::uint64_t> rf(2u * n), rg(2u * n);
        std::vector<std::uint64_t> sf(2u * n), sg(2u * n);
        fill_limbs(a, rng);
        fill_limbs(b, rng);

        if (!fft::mul_auto(rf.data(), a.data(), n, b.data(), n)) {
            std::printf("mul_auto returned 0 at n=%d\n", n);
            return 1;
        }
        mpn_mul_n(reinterpret_cast<mp_ptr>(rg.data()),
                  reinterpret_cast<mp_srcptr>(a.data()),
                  reinterpret_cast<mp_srcptr>(b.data()), n);
        int bad = first_mismatch(rf, rg);
        if (bad >= 0) {
            std::printf("mul mismatch n=%d limb=%d\n", n, bad);
            return 1;
        }

        if (!fft::sqr_auto(sf.data(), a.data(), n)) {
            std::printf("sqr_auto returned 0 at n=%d\n", n);
            return 1;
        }
        mpn_sqr(reinterpret_cast<mp_ptr>(sg.data()),
                reinterpret_cast<mp_srcptr>(a.data()), n);
        bad = first_mismatch(sf, sg);
        if (bad >= 0) {
            std::printf("sqr mismatch n=%d limb=%d\n", n, bad);
            return 1;
        }

        std::printf("ok n=%d\n", n);
    }

    const int pairs[][2] = {
        {17001, 20000}, {10000, 32768}, {16385, 32760}
    };
    for (const auto& p : pairs) {
        int an = p[0], bn = p[1];
        std::vector<std::uint64_t> a(an), b(bn);
        std::vector<std::uint64_t> rf(an + bn), rg(an + bn);
        fill_limbs(a, rng);
        fill_limbs(b, rng);

        if (!fft::mul_auto(rf.data(), a.data(), an, b.data(), bn)) {
            std::printf("mul_auto returned 0 at an=%d bn=%d\n", an, bn);
            return 1;
        }
        if (an >= bn)
            mpn_mul(reinterpret_cast<mp_ptr>(rg.data()),
                    reinterpret_cast<mp_srcptr>(a.data()), an,
                    reinterpret_cast<mp_srcptr>(b.data()), bn);
        else
            mpn_mul(reinterpret_cast<mp_ptr>(rg.data()),
                    reinterpret_cast<mp_srcptr>(b.data()), bn,
                    reinterpret_cast<mp_srcptr>(a.data()), an);
        int bad = first_mismatch(rf, rg);
        if (bad >= 0) {
            std::printf("mul mismatch an=%d bn=%d limb=%d\n", an, bn, bad);
            return 1;
        }
        std::printf("ok an=%d bn=%d\n", an, bn);
    }
}

// Sweep benchmark for fft::mul_auto trunk-width delegation.
//
// The sweep is transform-length driven at 1/8 octave by default. For each
// target transform length, it picks a balanced n x n limb size that lands in
// the corresponding trunk-width band, then records the actual transform chosen
// by the FFT size picker.
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 -I../src \
//           bench_trunk_delegate.cpp ../src/fft.cpp -lgmp -lm -o bench_trunk_delegate
//
// Args: [--min-transform N] [--max-transform N] [--factor F] [--seconds S]
//       [--repeats N] [--warmups N] [--seed N] [--csv PATH]
//       [--mpn-until-limbs N]

#include "fft.hpp"

#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct bench_config {
    std::uint32_t min_transform = 8;
    std::uint32_t max_transform = 1u << 23;
    double factor = 1.09050773266525765;  // 2^(1/8): 8 points/octave.
    double seconds = 0.002;
    unsigned warmups = 0;
    unsigned repeats = 1;
    std::uint64_t seed = 0xC0D3F00Dull;
    std::ptrdiff_t mpn_until_limbs = 131072;
    const char* csv_path = nullptr;
};

struct size_info {
    unsigned N;
    unsigned n_branch;
    unsigned M;
    unsigned trunk_bits;
};

static unsigned ceil_pow2_u(unsigned x) {
    unsigned r = 1;
    while (r < x) r <<= 1;
    return r;
}

static void pick_size(unsigned needed, unsigned& N, unsigned& n_br, unsigned& M) {
    N = ceil_pow2_u(needed);
    n_br = N;
    M = 1;
    for (unsigned m : { 3u, 5u, 7u }) {
        unsigned nb = ceil_pow2_u((needed + m - 1u) / m);
        unsigned Nf = m * nb;
        unsigned min_nb = (m == 7u) ? 2048u : 4u;
        if (nb >= min_nb && Nf >= needed && Nf < N) {
            N = Nf;
            n_br = nb;
            M = m;
        }
    }
}

static unsigned max_transform_for_bits(unsigned bits) {
    if (bits >= 16u) return 1u << 16;
    if (bits == 15u) return 1u << 19;
    if (bits == 14u) return 1u << 21;
    return 1u << 23;
}

static unsigned bits_for_target_transform(unsigned target) {
    if (target <= (1u << 16)) return 16;
    if (target <= (1u << 19)) return 15;
    if (target <= (1u << 21)) return 14;
    return 13;
}

static bool delegate_size(std::ptrdiff_t limbs, size_info& out) {
    for (unsigned bits : { 16u, 15u, 14u, 13u }) {
        std::uint64_t digits64 = (64u * std::uint64_t(limbs) + bits - 1u) / bits;
        if (digits64 > 0xffffffffULL) return false;

        unsigned N, n_branch, M;
        pick_size(unsigned(digits64), N, n_branch, M);
        if (N <= max_transform_for_bits(bits)) {
            out = { N, n_branch, M, bits };
            return true;
        }
    }
    return false;
}

static std::ptrdiff_t limb_limit_for_transform(unsigned transform, unsigned bits) {
    return std::ptrdiff_t((std::uint64_t(transform) * bits) / 64u);
}

static std::ptrdiff_t max_limb_limit_for_sweep(unsigned max_transform) {
    std::ptrdiff_t max_limbs =
        limb_limit_for_transform(max_transform, bits_for_target_transform(max_transform));
    for (unsigned bits : { 16u, 15u, 14u, 13u }) {
        unsigned band_max = max_transform_for_bits(bits);
        if (band_max <= max_transform)
            max_limbs = std::max(max_limbs, limb_limit_for_transform(band_max, bits));
    }
    return max_limbs > 0 ? max_limbs : 1;
}

static double bench_now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_limbs(std::uint64_t* p, std::ptrdiff_t n, std::mt19937_64& rng) {
    for (std::ptrdiff_t i = 0; i < n; ++i) p[i] = rng();
    if (n > 0) p[n - 1] |= 1ULL << 63;
}

template <class F>
static double time_one(F run, unsigned repeats, unsigned warmups, double seconds) {
    for (unsigned i = 0; i < warmups; ++i) run();
    std::vector<double> times;
    times.reserve(repeats);
    for (unsigned rep = 0; rep < repeats; ++rep) {
        unsigned iters = 0;
        double start = bench_now();
        double elapsed = 0.0;
        do {
            run();
            ++iters;
            elapsed = bench_now() - start;
        } while (elapsed < seconds);
        times.push_back(elapsed * 1e9 / iters);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--min-transform N] [--max-transform N] [--factor F] "
                 "[--seconds S] [--repeats N] [--warmups N] [--seed N] "
                 "[--csv PATH] [--mpn-until-limbs N]\n",
                 argv0);
}

static bool parse_u64(const char* s, std::uint64_t& out) {
    char* end;
    out = std::strtoull(s, &end, 0);
    return *s != 0 && *end == 0;
}

static bool parse_double(const char* s, double& out) {
    char* end;
    out = std::strtod(s, &end);
    return *s != 0 && *end == 0;
}

int main(int argc, char** argv) {
    bench_config cfg;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](int iarg) {
            if (iarg >= argc) {
                usage(argv[0]);
                std::exit(1);
            }
        };
        if (!std::strcmp(argv[i], "--min-transform")) {
            need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.min_transform = unsigned(v);
        } else if (!std::strcmp(argv[i], "--max-transform")) {
            need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.max_transform = unsigned(v);
        } else if (!std::strcmp(argv[i], "--factor")) {
            need(++i); parse_double(argv[i], cfg.factor);
        } else if (!std::strcmp(argv[i], "--seconds")) {
            need(++i); parse_double(argv[i], cfg.seconds);
        } else if (!std::strcmp(argv[i], "--repeats")) {
            need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.repeats = unsigned(v);
        } else if (!std::strcmp(argv[i], "--warmups")) {
            need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.warmups = unsigned(v);
        } else if (!std::strcmp(argv[i], "--seed")) {
            need(++i); parse_u64(argv[i], cfg.seed);
        } else if (!std::strcmp(argv[i], "--csv")) {
            need(++i); cfg.csv_path = argv[i];
        } else if (!std::strcmp(argv[i], "--mpn-until-limbs")) {
            need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.mpn_until_limbs = std::ptrdiff_t(v);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (cfg.min_transform < 2u || cfg.max_transform < cfg.min_transform ||
        cfg.max_transform > (1u << 23) || cfg.factor <= 1.0 ||
        cfg.seconds < 0.0 || cfg.repeats == 0) {
        usage(argv[0]);
        return 1;
    }

    std::ptrdiff_t max_limbs = max_limb_limit_for_sweep(cfg.max_transform);

    std::mt19937_64 rng(cfg.seed);
    std::vector<std::uint64_t> A(max_limbs), B(max_limbs);
    std::vector<std::uint64_t> R_fft(2 * max_limbs), R_mpn(2 * max_limbs);
    fill_limbs(A.data(), max_limbs, rng);
    fill_limbs(B.data(), max_limbs, rng);

    FILE* csv = cfg.csv_path ? std::fopen(cfg.csv_path, "w") : stdout;
    if (!csv) {
        std::perror(cfg.csv_path);
        return 1;
    }

    std::fprintf(stderr,
                 "Trunk delegate sweep: min_N=%u max_N=%u factor=%.6f seconds=%.4f "
                 "repeats=%u warmups=%u mpn_until_limbs=%td seed=0x%llx\n",
                 cfg.min_transform, cfg.max_transform, cfg.factor, cfg.seconds,
                 cfg.repeats, cfg.warmups, cfg.mpn_until_limbs,
                 (unsigned long long)cfg.seed);

    std::fprintf(csv,
                 "target_N,limbs,trunk_bits,fft_N,fft_branch,fft_M,"
                 "fft_ns,fft_ns_per_limb,mpn_ns,mpn_ns_per_limb,speedup\n");

    std::ptrdiff_t prev_limbs = 0;
    for (double target_d = double(cfg.min_transform);
         target_d <= double(cfg.max_transform) * 1.0000001;
         target_d *= cfg.factor) {
        unsigned target = unsigned(target_d + 0.5);
        if (target > cfg.max_transform) target = cfg.max_transform;

        unsigned target_bits = bits_for_target_transform(target);
        std::ptrdiff_t limbs =
            std::ptrdiff_t((std::uint64_t(target) * target_bits) / 64u);
        if (limbs < 1) limbs = 1;
        if (limbs == prev_limbs && target != cfg.max_transform)
            continue;
        prev_limbs = limbs;

        size_info si;
        if (!delegate_size(limbs, si)) {
            std::fprintf(stderr, "delegate picker failed at limbs=%td target_N=%u\n",
                         limbs, target);
            return 1;
        }

        if (!fft::mul_auto(R_fft.data(), A.data(), limbs, B.data(), limbs)) {
            std::fprintf(stderr, "fft::mul_auto returned 0 at limbs=%td\n", limbs);
            return 1;
        }

        bool have_mpn = limbs <= cfg.mpn_until_limbs;
        double mpn_ns = 0.0;
        if (have_mpn) {
            mpn_mul_n((mp_ptr)R_mpn.data(), (mp_srcptr)A.data(),
                      (mp_srcptr)B.data(), limbs);
            for (std::ptrdiff_t i = 0; i < 2 * limbs; ++i) {
                if (R_fft[i] != R_mpn[i]) {
                    std::fprintf(stderr, "mismatch at limbs=%td limb=%td\n", limbs, i);
                    return 1;
                }
            }
            mpn_ns = time_one(
                [&] {
                    mpn_mul_n((mp_ptr)R_mpn.data(), (mp_srcptr)A.data(),
                              (mp_srcptr)B.data(), limbs);
                },
                cfg.repeats, cfg.warmups, cfg.seconds);
        }

        double fft_ns = time_one(
            [&] {
                fft::mul_auto(R_fft.data(), A.data(), limbs, B.data(), limbs);
            },
            cfg.repeats, cfg.warmups, cfg.seconds);

        std::fprintf(csv, "%u,%td,%u,%u,%u,%u,%.1f,%.6f,",
                     target, limbs, si.trunk_bits, si.N, si.n_branch, si.M,
                     fft_ns, fft_ns / double(limbs));
        if (have_mpn) {
            std::fprintf(csv, "%.1f,%.6f,%.6f\n",
                         mpn_ns, mpn_ns / double(limbs), mpn_ns / fft_ns);
        } else {
            std::fprintf(csv, ",,\n");
        }

        std::fprintf(stderr,
                     "target_N=%8u limbs=%8td bits=%2u actual_N=%8u M=%u "
                     "fft=%10.1f ns (%7.3f ns/limb)",
                     target, limbs, si.trunk_bits, si.N, si.M,
                     fft_ns, fft_ns / double(limbs));
        if (have_mpn) {
            std::fprintf(stderr, " mpn=%10.1f ns speedup=%.2fx",
                         mpn_ns, mpn_ns / fft_ns);
        }
        std::fputc('\n', stderr);

        if (target == cfg.max_transform)
            break;
    }

    if (csv != stdout) std::fclose(csv);
    return 0;
}

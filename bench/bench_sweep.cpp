// Sweep benchmark: fft::mul (int_fft) vs mpn_mul_n (system libgmp dispatch).
// Emits CSV for plotting.
//
// Single-header build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 -DINT_FFT_SINGLE_HEADER \
//           -I.. bench_sweep.cpp -lgmp -lm -o bench_sweep
// Split build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//           -I../src bench_sweep.cpp ../src/fft.cpp -lgmp -lm -o bench_sweep
//
// Args: [--min N] [--max N] [--factor F] [--seconds S]
//       [--repeats N] [--warmups N] [--seed N] [--csv PATH]

#ifdef INT_FFT_SINGLE_HEADER
#  define INT_FFT_IMPLEMENTATION
#  include "fft.hpp"
#else
#  include "fft.hpp"
#endif

#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct bench_config {
    std::ptrdiff_t min_limbs = 16;
    std::ptrdiff_t max_limbs = 16384;
    double factor = 1.09050773266525765;  // 2^(1/8): 8 points/octave.
    double seconds = 0.02;
    unsigned warmups = 1;
    unsigned repeats = 3;
    std::uint64_t seed = 0xC0D3F00Dull;
    const char* csv_path = nullptr;
};

// Replicate choose_fft_size() for label reporting (can't link fft::internals).
static unsigned ceil_pow2_u(unsigned x) { unsigned r = 1; while (r < x) r <<= 1; return r; }
static void pick_size(unsigned needed, unsigned& N, unsigned& n_br, unsigned& M) {
    N = ceil_pow2_u(needed); n_br = N; M = 1;
    for (unsigned m : { 3u, 5u, 7u }) {
        unsigned nb = ceil_pow2_u((needed + m - 1u) / m);
        unsigned Nf = m * nb;
        unsigned min_nb = (m == 7u) ? 2048u : 4u;
        if (nb >= min_nb && Nf >= needed && Nf < N) { N = Nf; n_br = nb; M = m; }
    }
}

static double bench_now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void fill_limbs(std::uint64_t* p, std::ptrdiff_t n, std::mt19937_64& rng) {
    for (std::ptrdiff_t i = 0; i < n; ++i) p[i] = rng();
    if (n > 0) p[n - 1] |= 1ULL << 63;   // ensure full-width operand
}

static bool run_fft(std::uint64_t* r,
                    const std::uint64_t* a, std::ptrdiff_t an,
                    const std::uint64_t* b, std::ptrdiff_t bn) {
    return fft::mul(r, a, an, b, bn) != 0;
}
static void run_mpn(mp_ptr r, mp_srcptr a, mp_srcptr b, mp_size_t n) {
    mpn_mul_n(r, a, b, n);
}

// Measure one algo: run until `seconds` elapsed, return median of `repeats`
// across that. Returns ns per call.
template <class F>
static double time_one(F run, unsigned repeats, unsigned warmups, double seconds) {
    for (unsigned i = 0; i < warmups; ++i) run();
    std::vector<double> times;
    times.reserve(repeats);
    for (unsigned rep = 0; rep < repeats; ++rep) {
        unsigned iters = 0;
        double start = bench_now();
        double elapsed;
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
                 "Usage: %s [--min N] [--max N] [--factor F] [--seconds S] "
                 "[--repeats N] [--warmups N] [--seed N] [--csv PATH]\n",
                 argv0);
}

static bool parse_u64(const char* s, std::uint64_t& out) {
    char* end; out = std::strtoull(s, &end, 0); return *s != 0 && *end == 0;
}
static bool parse_double(const char* s, double& out) {
    char* end; out = std::strtod(s, &end); return *s != 0 && *end == 0;
}

int main(int argc, char** argv) {
    bench_config cfg;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](int i){ if (i >= argc) { usage(argv[0]); std::exit(1); } };
        if (!std::strcmp(argv[i], "--min"))      { need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.min_limbs = (std::ptrdiff_t)v; }
        else if (!std::strcmp(argv[i], "--max")) { need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.max_limbs = (std::ptrdiff_t)v; }
        else if (!std::strcmp(argv[i], "--factor")) { need(++i); parse_double(argv[i], cfg.factor); }
        else if (!std::strcmp(argv[i], "--seconds")) { need(++i); parse_double(argv[i], cfg.seconds); }
        else if (!std::strcmp(argv[i], "--repeats")) { need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.repeats = (unsigned)v; }
        else if (!std::strcmp(argv[i], "--warmups")) { need(++i); std::uint64_t v; parse_u64(argv[i], v); cfg.warmups = (unsigned)v; }
        else if (!std::strcmp(argv[i], "--seed"))    { need(++i); parse_u64(argv[i], cfg.seed); }
        else if (!std::strcmp(argv[i], "--csv"))     { need(++i); cfg.csv_path = argv[i]; }
        else { usage(argv[0]); return 1; }
    }
    if (cfg.min_limbs < 1 || cfg.max_limbs < cfg.min_limbs || cfg.factor <= 1.0 || cfg.seconds <= 0) {
        usage(argv[0]); return 1;
    }
    if (cfg.max_limbs > 16384) cfg.max_limbs = 16384;

    std::mt19937_64 rng(cfg.seed);
    std::vector<std::uint64_t> A(cfg.max_limbs), B(cfg.max_limbs);
    std::vector<std::uint64_t> R_fft(2 * cfg.max_limbs), R_mpn(2 * cfg.max_limbs);
    fill_limbs(A.data(), cfg.max_limbs, rng);
    fill_limbs(B.data(), cfg.max_limbs, rng);

    FILE* csv = cfg.csv_path ? std::fopen(cfg.csv_path, "w") : stdout;
    if (!csv) { std::perror(cfg.csv_path); return 1; }

    std::fprintf(stderr,
                 "Sweep: min=%td max=%td factor=%.6f seconds=%.3f repeats=%u "
                 "warmups=%u seed=0x%llx\n",
                 cfg.min_limbs, cfg.max_limbs, cfg.factor, cfg.seconds,
                 cfg.repeats, cfg.warmups, (unsigned long long)cfg.seed);

    std::fprintf(csv, "limbs,fft_ns,fft_ns_per_limb,mpn_ns,mpn_ns_per_limb,fft_N,fft_M\n");

    auto next_size = [&](std::ptrdiff_t n) {
        std::ptrdiff_t nx = (std::ptrdiff_t)(double(n) * cfg.factor + 0.5);
        return nx > n ? nx : n + 1;
    };

    for (std::ptrdiff_t n = cfg.min_limbs; n <= cfg.max_limbs; n = next_size(n)) {
        // Sanity check: fft result must match mpn for this size.
        run_mpn((mp_ptr)R_mpn.data(), (mp_srcptr)A.data(), (mp_srcptr)B.data(), n);
        if (!run_fft(R_fft.data(), A.data(), n, B.data(), n)) {
            std::fprintf(stderr, "fft::mul returned 0 at n=%td\n", n); return 1;
        }
        for (std::ptrdiff_t i = 0; i < 2 * n; ++i) {
            if (R_fft[i] != R_mpn[i]) {
                std::fprintf(stderr, "mismatch at n=%td limb %td\n", n, i);
                return 1;
            }
        }

        double fft_ns = time_one(
            [&]{ run_fft(R_fft.data(), A.data(), n, B.data(), n); },
            cfg.repeats, cfg.warmups, cfg.seconds);
        double mpn_ns = time_one(
            [&]{ run_mpn((mp_ptr)R_mpn.data(), (mp_srcptr)A.data(), (mp_srcptr)B.data(), n); },
            cfg.repeats, cfg.warmups, cfg.seconds);

        unsigned N, n_br, M;
        pick_size(4u * (unsigned)n, N, n_br, M);

        std::fprintf(csv, "%td,%.1f,%.3f,%.1f,%.3f,%u,%u\n",
                     n, fft_ns, fft_ns / double(n),
                     mpn_ns, mpn_ns / double(n),
                     N, M);
        std::fprintf(stderr,
                     "n=%6td  fft=%9.1f ns (%6.2f ns/limb)  mpn=%9.1f ns (%6.2f ns/limb)  "
                     "ratio=%.2fx  N=%u M=%u\n",
                     n, fft_ns, fft_ns / double(n),
                     mpn_ns, mpn_ns / double(n),
                     mpn_ns / fft_ns, N, M);

        if (n == cfg.max_limbs) break;
    }

    if (csv != stdout) std::fclose(csv);
    return 0;
}

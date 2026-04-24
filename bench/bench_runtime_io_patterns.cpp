// White-box benchmark for RuntimeBitsCodec input loads under FFT access patterns.
//
// Build from the repository root:
//   clang++ -O3 -mavx2 -mfma -std=c++17 -Wall -Wextra -pedantic \
//       bench/bench_runtime_io_patterns.cpp -lm -o /tmp/bench_runtime_io_patterns
//
// The benchmark includes src/fft.cpp intentionally so it can exercise the
// internal encoded-address load API with the same radix-4 and PFA radix-3/5/7
// input patterns used by the FFT.

#include "../src/fft.cpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct bench_config {
    std::uint32_t pairs = 1u << 20;
    double seconds = 0.01;
    unsigned warmups = 1;
    unsigned repeats = 5;
    std::uint64_t seed = 0x510015ull;
    bool all_bits = false;
    const char* csv_path = nullptr;
};

volatile std::uint64_t g_sink = 0;

static double now_seconds()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::uint32_t floor_pow2(std::uint32_t x)
{
    std::uint32_t r = 1;
    while ((r << 1u) != 0u && (r << 1u) <= x) r <<= 1u;
    return r;
}

static void fill_limbs(std::vector<std::uint64_t>& src, std::mt19937_64& rng)
{
    for (std::uint64_t& x : src) x = rng();
}

static void fold_cv(cv x)
{
    alignas(32) double out[8];
    store(out, x);
    std::uint64_t fold = 0;
    for (double v : out)
        fold ^= std::uint64_t(v);
    g_sink ^= fold;
}

static bool same_cv(cv a, cv b)
{
    alignas(32) double aa[8], bb[8];
    store(aa, a);
    store(bb, b);
    for (unsigned i = 0; i < 8u; ++i) {
        if (aa[i] != bb[i])
            return false;
    }
    return true;
}

template <class F>
static double time_one(F run, unsigned repeats, unsigned warmups, double seconds)
{
    for (unsigned i = 0; i < warmups; ++i) run();
    std::vector<double> times;
    times.reserve(repeats);
    for (unsigned rep = 0; rep < repeats; ++rep) {
        unsigned iters = 0;
        double start = now_seconds();
        double elapsed = 0.0;
        do {
            run();
            ++iters;
            elapsed = now_seconds() - start;
        } while (elapsed < seconds);
        times.push_back(elapsed * 1e9 / iters);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static std::uint32_t encoded_pair_from_trunk(std::uint32_t trunk_index,
                                             unsigned tag)
{
    return trunk_index | tag;
}

static cv run_radix4_pattern(const std::uint64_t* src, std::size_t limb_count,
                             const RuntimeBitsCodec& io, std::uint32_t n)
{
    io.reset_input_cache();
    cv acc = { zero4(), zero4() };
    std::uint32_t l = n >> 2;
    for (std::uint32_t j = 0; j < l; j += 4u) {
        acc = acc + io.load_pair4_encoded(src, limb_count,
            encoded_pair_from_trunk(2u * (j + 0u * l), 0u));
        acc = acc + io.load_pair4_encoded(src, limb_count,
            encoded_pair_from_trunk(2u * (j + 1u * l), 1u));
        acc = acc + io.load_pair4_encoded(src, limb_count,
            encoded_pair_from_trunk(2u * (j + 2u * l), 2u));
        acc = acc + io.load_pair4_encoded(src, limb_count,
            encoded_pair_from_trunk(2u * (j + 3u * l), 3u));
    }
    return acc;
}

template <std::uint32_t M>
static cv run_pfa_pattern(const std::uint64_t* src, std::size_t limb_count,
                          const RuntimeBitsCodec& io, std::uint32_t n)
{
    static_assert(M == 3u || M == 5u || M == 7u, "PFA pattern must be 3, 5, or 7");
    io.reset_input_cache();
    cv acc = { zero4(), zero4() };

    alignas(32) std::uint32_t q_init[8] = {};
    std::uint32_t step = n % M;
    std::uint32_t a = 0;
    for (std::uint32_t m = 0; m < M; ++m) {
        q_init[a] = 2u * m * n + m;
        a += step;
        if (a >= M) a -= M;
    }

    __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(q_init));
    const __m256i rot = rot_idx_v<M>();
    const __m256i bump = _mm256_set1_epi32(8);

    for (std::uint32_t t = 0; t < (n >> 2); ++t) {
        alignas(32) std::uint32_t q[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(q), v);
        for (std::uint32_t i = 0; i < M; ++i)
            acc = acc + io.load_pair4_encoded(src, limb_count, q[i]);
        v = _mm256_add_epi32(v, bump);
        v = _mm256_permutevar8x32_epi32(v, rot);
    }
    return acc;
}

template <class Pattern>
static bool validate_pattern(const std::uint64_t* src, std::size_t limb_count,
                             const RuntimeBitsCodec& io, Pattern encoded_at,
                             std::uint32_t load_count)
{
    RuntimeBitsCodec cached(io.trunk_bits);
    for (std::uint32_t i = 0; i < load_count; ++i) {
        std::uint32_t encoded = encoded_at(i);
        std::uint32_t pair_index = (encoded & ~7u) >> 1;
        cv a = cached.load_pair4_encoded(src, limb_count, encoded);
        cv b = io.load_pair4_direct(src, limb_count, pair_index);
        if (!same_cv(a, b))
            return false;
    }
    return true;
}

static bool validate_radix4(const std::uint64_t* src, std::size_t limb_count,
                            const RuntimeBitsCodec& io, std::uint32_t n)
{
    std::uint32_t l = n >> 2;
    auto encoded_at = [=](std::uint32_t idx) {
        std::uint32_t j = (idx >> 2) * 4u;
        std::uint32_t h = idx & 3u;
        return encoded_pair_from_trunk(2u * (j + h * l), h);
    };
    return validate_pattern(src, limb_count, io, encoded_at, n >> 2);
}

template <std::uint32_t M>
static bool validate_pfa(const std::uint64_t* src, std::size_t limb_count,
                         const RuntimeBitsCodec& io, std::uint32_t n)
{
    std::vector<std::uint32_t> encoded;
    encoded.reserve((std::size_t(M) * n) >> 2);

    alignas(32) std::uint32_t q_init[8] = {};
    std::uint32_t step = n % M;
    std::uint32_t a = 0;
    for (std::uint32_t m = 0; m < M; ++m) {
        q_init[a] = 2u * m * n + m;
        a += step;
        if (a >= M) a -= M;
    }
    __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(q_init));
    const __m256i rot = rot_idx_v<M>();
    const __m256i bump = _mm256_set1_epi32(8);
    for (std::uint32_t t = 0; t < (n >> 2); ++t) {
        alignas(32) std::uint32_t q[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(q), v);
        for (std::uint32_t i = 0; i < M; ++i)
            encoded.push_back(q[i]);
        v = _mm256_add_epi32(v, bump);
        v = _mm256_permutevar8x32_epi32(v, rot);
    }
    auto encoded_at = [&](std::uint32_t i) { return encoded[i]; };
    return validate_pattern(src, limb_count, io, encoded_at,
                            std::uint32_t(encoded.size()));
}

static bool validate_all()
{
    std::mt19937_64 rng(0x51EEDull);
    for (unsigned bits = 1; bits < 16u; ++bits) {
        RuntimeBitsCodec io(bits);
        std::uint32_t branch = 1024u;
        std::uint32_t max_pairs = 7u * branch;
        std::size_t limbs = (std::size_t(max_pairs) * io.pair_bits + 63u) >> 6;
        std::vector<std::uint64_t> src(limbs + 4u);
        fill_limbs(src, rng);
        if (!validate_radix4(src.data(), src.size(), io, branch) ||
            !validate_pfa<3>(src.data(), src.size(), io, branch) ||
            !validate_pfa<5>(src.data(), src.size(), io, branch) ||
            !validate_pfa<7>(src.data(), src.size(), io, branch)) {
            std::fprintf(stderr, "pattern validation failed for bits=%u\n", bits);
            return false;
        }
    }
    return true;
}

struct pattern_case {
    const char* name;
    std::uint32_t M;
    std::uint32_t branch_pairs;
    std::uint32_t total_pairs;
    std::uint32_t tile_loads;
};

static std::uint32_t branch_for(std::uint32_t pairs, std::uint32_t M)
{
    std::uint32_t branch = floor_pow2(pairs / M);
    if (branch < 1024u) branch = 1024u;
    branch &= ~3u;
    return branch;
}

static void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--pairs N] [--seconds S] [--repeats N] "
                 "[--warmups N] [--seed N] [--all-bits] [--csv PATH]\n",
                 argv0);
}

static bool parse_u64(const char* s, std::uint64_t& out)
{
    char* end;
    out = std::strtoull(s, &end, 0);
    return *s != 0 && *end == 0;
}

static bool parse_double(const char* s, double& out)
{
    char* end;
    out = std::strtod(s, &end);
    return *s != 0 && *end == 0;
}

} // namespace

int main(int argc, char** argv)
{
    bench_config cfg;
    cfg.seed = 0x510015ull;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](int iarg) {
            if (iarg >= argc) {
                usage(argv[0]);
                std::exit(1);
            }
        };
        if (!std::strcmp(argv[i], "--pairs")) {
            need(++i);
            std::uint64_t v;
            if (!parse_u64(argv[i], v)) return 1;
            cfg.pairs = std::uint32_t(v);
        } else if (!std::strcmp(argv[i], "--seconds")) {
            need(++i);
            if (!parse_double(argv[i], cfg.seconds)) return 1;
        } else if (!std::strcmp(argv[i], "--repeats")) {
            need(++i);
            std::uint64_t v;
            if (!parse_u64(argv[i], v)) return 1;
            cfg.repeats = unsigned(v);
        } else if (!std::strcmp(argv[i], "--warmups")) {
            need(++i);
            std::uint64_t v;
            if (!parse_u64(argv[i], v)) return 1;
            cfg.warmups = unsigned(v);
        } else if (!std::strcmp(argv[i], "--seed")) {
            need(++i);
            if (!parse_u64(argv[i], cfg.seed)) return 1;
        } else if (!std::strcmp(argv[i], "--all-bits")) {
            cfg.all_bits = true;
        } else if (!std::strcmp(argv[i], "--csv")) {
            need(++i);
            cfg.csv_path = argv[i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (cfg.pairs < 8192u || cfg.repeats == 0u || cfg.seconds < 0.0) {
        usage(argv[0]);
        return 1;
    }
    if (!validate_all())
        return 1;

    std::uint32_t b4 = branch_for(cfg.pairs, 1u);
    pattern_case cases[] = {
        { "radix4", 4u, b4, b4, b4 >> 2 },
        { "radix3", 3u, branch_for(cfg.pairs, 3u), 0u, 0u },
        { "radix5", 5u, branch_for(cfg.pairs, 5u), 0u, 0u },
        { "radix7", 7u, branch_for(cfg.pairs, 7u), 0u, 0u },
    };
    for (pattern_case& c : cases) {
        if (c.M != 4u) {
            c.total_pairs = c.M * c.branch_pairs;
            c.tile_loads = c.total_pairs >> 2;
        }
    }
    std::uint32_t max_total_pairs = 0;
    for (const pattern_case& c : cases)
        max_total_pairs = std::max(max_total_pairs, c.total_pairs);

    FILE* csv = cfg.csv_path ? std::fopen(cfg.csv_path, "w") : stdout;
    if (!csv) {
        std::perror(cfg.csv_path);
        return 1;
    }
    std::fprintf(stderr,
                 "RBC pattern bench: max_pairs=%u seconds=%.4f repeats=%u "
                 "warmups=%u seed=0x%llx\n",
                 cfg.pairs, cfg.seconds, cfg.repeats, cfg.warmups,
                 (unsigned long long)cfg.seed);
    std::fprintf(csv,
                 "bits,pattern,M,branch_pairs,total_pairs,tile_loads,"
                 "ns_per_tile,ns_per_pair,total_ns,sink\n");

    std::vector<unsigned> widths;
    if (cfg.all_bits) {
        for (unsigned bits = 1; bits < 16u; ++bits) widths.push_back(bits);
    } else {
        widths = { 15u, 14u, 13u };
    }

    std::mt19937_64 rng(cfg.seed);
    for (unsigned bits : widths) {
        RuntimeBitsCodec io(bits);
        std::size_t max_limbs =
            (std::size_t(max_total_pairs) * io.pair_bits + 63u) >> 6;
        std::vector<std::uint64_t> src(max_limbs + 8u);
        fill_limbs(src, rng);

        for (const pattern_case& c : cases) {
            auto run = [&] {
                cv acc = { zero4(), zero4() };
                if (!std::strcmp(c.name, "radix4")) {
                    acc = run_radix4_pattern(src.data(), src.size(), io,
                                             c.branch_pairs);
                } else if (!std::strcmp(c.name, "radix3")) {
                    acc = run_pfa_pattern<3>(src.data(), src.size(), io,
                                             c.branch_pairs);
                } else if (!std::strcmp(c.name, "radix5")) {
                    acc = run_pfa_pattern<5>(src.data(), src.size(), io,
                                             c.branch_pairs);
                } else {
                    acc = run_pfa_pattern<7>(src.data(), src.size(), io,
                                             c.branch_pairs);
                }
                fold_cv(acc);
            };

            double ns = time_one(run, cfg.repeats, cfg.warmups, cfg.seconds);
            std::fprintf(csv, "%u,%s,%u,%u,%u,%u,%.6f,%.6f,%.3f,%llu\n",
                         bits, c.name, c.M, c.branch_pairs, c.total_pairs,
                         c.tile_loads, ns / c.tile_loads, ns / c.total_pairs,
                         ns, (unsigned long long)g_sink);
            std::fflush(csv);
        }
    }

    if (csv != stdout)
        std::fclose(csv);
    return 0;
}

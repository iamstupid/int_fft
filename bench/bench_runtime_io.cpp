// White-box benchmark for runtime-width input I/O.
//
// Build from the repository root:
//   clang++ -O3 -mavx2 -mfma -std=c++17 -Wall -Wextra -pedantic \
//       bench/bench_runtime_io.cpp -lm -o /tmp/bench_runtime_io
//
// This includes src/fft.cpp intentionally so the benchmark can validate and
// time the internal RuntimeBitsCodec without adding public benchmarking hooks.

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
    unsigned repeats = 3;
    std::uint64_t seed = 0x105B17E5ull;
    bool all_bits = false;
    const char* csv_path = nullptr;
};

struct AosoVCache {
    alignas(32) std::uint32_t tile[RuntimeBitsCodec::input_cache_tiles][8];
};

volatile std::uint64_t g_sink = 0;

FFT_INLINE void bench_escape(const void* p)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(p) : "memory");
#else
    g_sink ^= reinterpret_cast<std::uintptr_t>(p);
#endif
}

static void store_groups(AosoVCache& cache, std::uint32_t tile,
                         std::uint64_t g0, std::uint64_t g1,
                         const RuntimeBitsCodec& io)
{
    std::uint64_t p0 = g0 & io.pair_mask;
    std::uint64_t p1 = g0 >> io.pair_bits;
    std::uint64_t p2 = g1 & io.pair_mask;
    std::uint64_t p3 = g1 >> io.pair_bits;

    std::uint32_t* q = cache.tile[tile];
    q[0] = std::uint32_t(p0 & io.trunk_mask);
    q[1] = std::uint32_t(p1 & io.trunk_mask);
    q[2] = std::uint32_t(p2 & io.trunk_mask);
    q[3] = std::uint32_t(p3 & io.trunk_mask);
    q[4] = std::uint32_t((p0 >> io.trunk_bits) & io.trunk_mask);
    q[5] = std::uint32_t((p1 >> io.trunk_bits) & io.trunk_mask);
    q[6] = std::uint32_t((p2 >> io.trunk_bits) & io.trunk_mask);
    q[7] = std::uint32_t((p3 >> io.trunk_bits) & io.trunk_mask);
}

static void refill_independent(const std::uint64_t* src, std::size_t limb_count,
                               const RuntimeBitsCodec& io,
                               std::uint32_t base_pair, AosoVCache& cache)
{
    std::size_t bit = std::size_t(base_pair) * io.pair_bits;
    for (std::uint32_t t = 0; t < RuntimeBitsCodec::input_cache_tiles; ++t) {
        std::uint64_t g0 = load_bits_u64(src, limb_count, bit, io.group_bits);
        bit += io.group_bits;
        std::uint64_t g1 = load_bits_u64(src, limb_count, bit, io.group_bits);
        bit += io.group_bits;
        store_groups(cache, t, g0, g1, io);
    }
}

static void refill_stream(const std::uint64_t* src, std::size_t limb_count,
                          const RuntimeBitsCodec& io,
                          std::uint32_t base_pair, AosoVCache& cache)
{
    std::size_t base_bit = std::size_t(base_pair) * io.pair_bits;
    std::size_t base_limb = base_bit >> 6;
    RuntimeBitStream bits(src, limb_count, base_limb);

    for (std::uint32_t t = 0; t < RuntimeBitsCodec::input_cache_tiles; ++t) {
        std::uint64_t g0 = bits.take(io.group_bits);
        std::uint64_t g1 = bits.take(io.group_bits);
        store_groups(cache, t, g0, g1, io);
    }
}

static bool same_cache(const AosoVCache& a, const AosoVCache& b)
{
    return std::memcmp(&a, &b, sizeof(AosoVCache)) == 0;
}

static bool validate_one_load(const RuntimeBitsCodec& io,
                              const AosoVCache& expected,
                              const std::uint64_t* src,
                              std::size_t limb_count,
                              unsigned tag,
                              std::uint32_t base_pair,
                              std::uint32_t tile)
{
    std::uint32_t pair = base_pair + 4u * tile;
    cv x = io.load_pair4_encoded(src, limb_count, (2u * pair) | tag);
    alignas(32) double out[8];
    store(out, x);
    for (std::uint32_t i = 0; i < 8u; ++i) {
        if (out[i] != double(expected.tile[tile][i]))
            return false;
    }
    return true;
}

static bool validate()
{
    std::mt19937_64 rng(0xD15EA5E5ull);
    for (unsigned bits = 1; bits < 16u; ++bits) {
        RuntimeBitsCodec io(bits);
        for (unsigned trial = 0; trial < 200u; ++trial) {
            std::uint32_t base_pair = 64u * std::uint32_t(rng() & 255u);
            std::size_t full_limbs =
                (std::size_t(base_pair + RuntimeBitsCodec::input_cache_pairs) *
                     io.pair_bits +
                 63u) >> 6;
            std::size_t limb_count = full_limbs == 0 ? 0 : std::size_t(rng() % (full_limbs + 1u));
            std::vector<std::uint64_t> src(full_limbs + 4u);
            for (std::uint64_t& x : src) x = rng();

            const std::size_t limb_cases[] = { limb_count, src.size() };
            for (std::size_t validated_limbs : limb_cases) {
                for (unsigned tag = 0; tag < 8u; ++tag) {
                    AosoVCache independent{};
                    AosoVCache stream{};
                    refill_independent(src.data(), validated_limbs, io, base_pair, independent);
                    refill_stream(src.data(), validated_limbs, io, base_pair, stream);
                    if (!same_cache(independent, stream)) {
                        std::fprintf(stderr,
                                     "cache mismatch: bits=%u trial=%u tag=%u base_pair=%u limbs=%zu\n",
                                     bits, trial, tag, base_pair, validated_limbs);
                        return false;
                    }

                    io.reset_input_cache();
                    io.refill_input_cache(src.data(), validated_limbs, tag, base_pair);
                    if (std::memcmp(io.input_cache[tag], independent.tile,
                                    sizeof(independent.tile)) != 0) {
                        std::fprintf(stderr,
                                     "codec refill mismatch: bits=%u trial=%u tag=%u base_pair=%u limbs=%zu\n",
                                     bits, trial, tag, base_pair, validated_limbs);
                        return false;
                    }

                    io.reset_input_cache();
                    if (!validate_one_load(io, independent, src.data(), validated_limbs, tag,
                                           base_pair, std::uint32_t(rng() % RuntimeBitsCodec::input_cache_tiles))) {
                        std::fprintf(stderr,
                                     "codec load mismatch: bits=%u trial=%u tag=%u base_pair=%u limbs=%zu\n",
                                     bits, trial, tag, base_pair, validated_limbs);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static double bench_now()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
static double time_one(F run, unsigned repeats, unsigned warmups, double seconds)
{
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

static void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "Usage: %s [--pairs N] [--seconds S] [--repeats N] [--warmups N] "
                 "[--seed N] [--all-bits] [--csv PATH]\n",
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

static void fill_limbs(std::vector<std::uint64_t>& src, std::mt19937_64& rng)
{
    for (std::uint64_t& x : src) x = rng();
}

static double bench_independent(const std::vector<std::uint64_t>& src,
                                const RuntimeBitsCodec& io,
                                std::uint32_t segments,
                                const bench_config& cfg)
{
    AosoVCache cache{};
    auto run = [&] {
        for (std::uint32_t s = 0; s < segments; ++s) {
            refill_independent(src.data(), src.size(), io,
                               s * RuntimeBitsCodec::input_cache_pairs, cache);
            bench_escape(&cache);
        }
    };
    return time_one(run, cfg.repeats, cfg.warmups, cfg.seconds);
}

static double bench_stream(const std::vector<std::uint64_t>& src,
                           const RuntimeBitsCodec& io,
                           std::uint32_t segments,
                           const bench_config& cfg)
{
    AosoVCache cache{};
    auto run = [&] {
        for (std::uint32_t s = 0; s < segments; ++s) {
            refill_stream(src.data(), src.size(), io,
                          s * RuntimeBitsCodec::input_cache_pairs, cache);
            bench_escape(&cache);
        }
    };
    return time_one(run, cfg.repeats, cfg.warmups, cfg.seconds);
}

static double bench_codec_refill(const std::vector<std::uint64_t>& src,
                                 RuntimeBitsCodec& io,
                                 std::uint32_t segments,
                                 const bench_config& cfg)
{
    auto run = [&] {
        io.reset_input_cache();
        for (std::uint32_t s = 0; s < segments; ++s) {
            io.refill_input_cache(src.data(), src.size(), 0u,
                                  s * RuntimeBitsCodec::input_cache_pairs);
            bench_escape(io.input_cache[0]);
        }
    };
    return time_one(run, cfg.repeats, cfg.warmups, cfg.seconds);
}

static double bench_codec_load(const std::vector<std::uint64_t>& src,
                               RuntimeBitsCodec& io,
                               std::uint32_t pairs,
                               const bench_config& cfg)
{
    auto run = [&] {
        io.reset_input_cache();
        cv acc = { zero4(), zero4() };
        for (std::uint32_t pair = 0; pair < pairs; pair += 4u) {
            cv x = io.load_pair4_encoded(src.data(), src.size(), 2u * pair);
            acc = acc + x;
        }
        alignas(32) double out[8];
        store(out, acc);
        std::uint64_t fold = 0;
        for (double x : out)
            fold ^= std::uint64_t(x);
        g_sink ^= fold;
    };
    return time_one(run, cfg.repeats, cfg.warmups, cfg.seconds);
}

} // namespace

int main(int argc, char** argv)
{
    bench_config cfg;
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

    if (cfg.pairs < RuntimeBitsCodec::input_cache_pairs || cfg.seconds < 0.0 ||
        cfg.repeats == 0u) {
        usage(argv[0]);
        return 1;
    }

    if (!validate())
        return 1;

    cfg.pairs =
        (cfg.pairs + RuntimeBitsCodec::input_cache_pairs - 1u) &
        ~(RuntimeBitsCodec::input_cache_pairs - 1u);
    std::uint32_t segments = cfg.pairs / RuntimeBitsCodec::input_cache_pairs;

    FILE* csv = cfg.csv_path ? std::fopen(cfg.csv_path, "w") : stdout;
    if (!csv) {
        std::perror(cfg.csv_path);
        return 1;
    }

    std::fprintf(stderr,
                 "Runtime IO bench: pairs=%u segments=%u seconds=%.4f repeats=%u "
                 "warmups=%u seed=0x%llx\n",
                 cfg.pairs, segments, cfg.seconds, cfg.repeats, cfg.warmups,
                 (unsigned long long)cfg.seed);
    std::fprintf(csv,
                 "bits,pairs,segments,independent_refill_ns_per_pair,"
                 "generic_stream_refill_ns_per_pair,codec_refill_ns_per_pair,"
                 "codec_refill_speedup,"
                 "codec_cached_load_ns_per_tile,sink\n");

    std::vector<unsigned> widths;
    if (cfg.all_bits) {
        for (unsigned bits = 1; bits < 16u; ++bits) widths.push_back(bits);
    } else {
        widths = { 15u, 14u, 13u };
    }

    std::mt19937_64 rng(cfg.seed);
    for (unsigned bits : widths) {
        RuntimeBitsCodec io(bits);
        std::size_t limb_count =
            (std::size_t(cfg.pairs) * io.pair_bits + 63u) >> 6;
        std::vector<std::uint64_t> src(limb_count + 4u);
        fill_limbs(src, rng);

        double independent_ns = bench_independent(src, io, segments, cfg);
        double stream_ns = bench_stream(src, io, segments, cfg);
        double codec_refill_ns = bench_codec_refill(src, io, segments, cfg);
        double codec_ns = bench_codec_load(src, io, cfg.pairs, cfg);

        double independent_per_pair = independent_ns / cfg.pairs;
        double stream_per_pair = stream_ns / cfg.pairs;
        double codec_refill_per_pair = codec_refill_ns / cfg.pairs;
        double codec_per_tile = codec_ns / (cfg.pairs / 4u);
        double speedup = codec_refill_ns > 0.0 ? independent_ns / codec_refill_ns : 0.0;

        std::fprintf(csv, "%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%llu\n",
                     bits, cfg.pairs, segments, independent_per_pair,
                     stream_per_pair, codec_refill_per_pair, speedup, codec_per_tile,
                     (unsigned long long)g_sink);
        std::fflush(csv);
    }

    if (csv != stdout)
        std::fclose(csv);
    return 0;
}

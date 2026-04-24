# Runtime Trunk I/O Design

## Goal

The FFT stages should stay independent of digit trunk width. The current fast
path uses 16-bit trunks packed naturally inside 64-bit limbs. Larger
multiplications need smaller trunks, likely below 16 bits, so the input and
output boundary code must support a runtime trunk width while preserving the
current 16-bit performance path.

Design targets:

- No FFT-stage boilerplate for alternate trunk widths.
- Template only I/O boundary code: input unpack, final inverse output emit, and
  partial normalization.
- Keep the 16-bit trunk path compile-time specialized and performance-neutral.
- Use runtime-decided trunk width for widths below 16 bits.
- Fuse runtime-width output with final inverse stores using trunk-dependent bit
  placement.

## Codec Boundary

Use a small codec concept. The FFT body continues to operate on `cv` tiles.

```cpp
struct U16Codec {
    using partial_type = unsigned __int128;
    static constexpr bool needs_partial_clear = false;

    FFT_INLINE cv load_pair4(const std::uint64_t* src,
                             std::size_t limbs,
                             std::uint32_t pair_index) const;

    FFT_INLINE void emit_tile(partial_type* partial,
                              std::size_t partial_n,
                              std::size_t tile_index,
                              cv x) const;

    int partial_to_limbs(std::uint64_t* rp,
                         partial_type* partial,
                         std::size_t result_limbs) const;
};

struct RuntimeBitsCodec {
    using partial_type = unsigned __int128;
    bool needs_partial_clear = true;
    unsigned trunk_bits;      // 1..15
    unsigned pair_bits;       // 2 * trunk_bits
    std::uint64_t trunk_mask;
    std::uint64_t pair_mask;

    FFT_INLINE cv load_pair4(const std::uint64_t* src,
                             std::size_t limbs,
                             std::uint32_t pair_index) const;

    FFT_INLINE void emit_tile(partial_type* partial,
                              std::size_t partial_n,
                              std::size_t tile_index,
                              cv x) const;

    int partial_to_limbs(std::uint64_t* rp,
                         partial_type* partial,
                         std::size_t result_limbs) const;
};
```

The templated call surface should be limited to:

```cpp
template <class Codec>
inline void unpack_fwd(..., const Codec& io);

template <std::uint32_t M, class Codec>
inline void fwd_pfa_unpack(..., const Codec& io);

template <class Codec>
inline void inv_final_range_to_partial(..., const Codec& io);

template <std::uint32_t M, class Codec>
inline void inv_pfa_butterfly_to_partial(..., const Codec& io);
```

All cascade, butterfly, pointwise, and PFA math stays unchanged.

## Input Loading

The existing 16-bit path remains a dedicated implementation. It should keep the
current `pshufb` plus widening sequence because the input is naturally aligned
as four 16-bit digits per limb.

For runtime widths below 16 bits, load two complex lanes at a time as one
packed pair group:

```text
pair_bits = 2 * trunk_bits
group_bits = 2 * pair_bits = 4 * trunk_bits <= 60

group0 = pairs 0 and 1
group1 = pairs 2 and 3
```

Each group is at most 60 bits, so each group load is at most two unaligned
64-bit word loads plus one shift/or/mask. For 15-bit trunks:

```text
group_bits = 60
pair_bits  = 30
pair = real_trunk + (imag_trunk << 15)
```

This is better than extracting every trunk independently because it reduces
boundary handling to two grouped bitfield loads per four complex lanes.

### Runtime Input Cache

Runtime input uses an encoded trunk address. Each `cv` input tile starts at an
8-trunk boundary, so the low three address bits are free for the stream tag:

```text
encoded    = trunk_index | stream_tag
stream_tag = encoded & 7
pair_index = (encoded & ~7) >> 1
```

The codec owns an eight-line input cache indexed by `stream_tag`:

```cpp
std::uint32_t max_pair[8];
alignas(32) std::uint32_t input_cache[8][16][8];
```

Each cache item is one AoSoV integer tile:

```text
[re0, re1, re2, re3, im0, im1, im2, im3]
```

The hot load is therefore one aligned 32-byte integer load plus two
`i32 -> f64` conversions:

```cpp
q    = vmovdqa input_cache[tag][tile]
re   = vcvtdq2pd low128(q)
imag = vcvtdq2pd high128(q)
```

Refill decodes a 64-pair segment into 16 AoSoV cache tiles. Segment starts are
64-pair aligned, so `base_pair * pair_bits` is always 64-bit-limb aligned.

For the hot widths 13, 14, and 15, each AoSoV tile consumes exactly `trunk_bits`
bytes:

```text
8 trunks * trunk_bits bits = trunk_bits bytes
```

The complete-segment refill therefore decodes one tile at a time with SIMD:

```text
load 16 bytes
vpshufb  -> eight 24-bit little-endian windows in u32 lanes
vpsrlvd  -> align the runtime-width trunks
vpand    -> mask to trunk width
vpermd   -> [re0, im0, re1, im1, ...] to [re0..re3, im0..im3]
vmovdqa  -> store the AoSoV u32 tile
```

This path needs one extra readable limb for the final tile's 16-byte load. Tail
segments that do not have that padding fall back to the fixed-width scalar
stream reader. The scalar fallback keeps a 64-bit residual buffer; because the
group width is at most 60 bits, a 128-bit stream register is not needed for
fixed 13/14/15-bit group reads.

This turns the four-head pow2 input pattern and the PFA multi-head pattern into
per-head sequential block decompression.

Pow2 forward unpack tags its four heads as `0..3`:

```text
j + 0*l -> tag 0
j + 1*l -> tag 1
j + 2*l -> tag 2
j + 3*l -> tag 3
```

PFA forward unpack stores the tag inside the rotating offset state:

```text
q_init[a] = 2 * m * n + m
bump      = 8
```

The low three tag bits survive both the `vpaddd` bump and the `vpermd`
rotation. Inverse output does not use this mechanism because it no longer reads
from runtime bit streams.

## Output Emit

The codec owns the conversion from final inverse `cv` tile to partial storage.

For `U16Codec`, `emit_tile` writes two already limb-aligned `u128` partials:

```text
pair = even + (odd << 16)
partial[2 * tile + 0] = pair0 + (pair1 << 32)
partial[2 * tile + 1] = pair2 + (pair3 << 32)
```

For `RuntimeBitsCodec`, `emit_tile` writes two sequential `u128` group partials:

```text
group_bits = 2 * pair_bits = 4 * trunk_bits
partial[2 * tile + 0] = pair0 + (pair1 << pair_bits)
partial[2 * tile + 1] = pair2 + (pair3 << pair_bits)
```

Unlike the 16-bit codec, these partials are spaced by `group_bits` rather than
64 output bits. The runtime path handles that bit offset sequentially in
`partial_to_limbs`, where it propagates carry in base `2^group_bits` and packs
the finalized chunks into 64-bit result limbs.

## Partial Normalization

`partial_to_limbs` belongs to the codec.

The 16-bit codec uses the existing carry chain over one partial per output limb.

The runtime-width codec normalizes bucketed partials. Since emit can add
misaligned pair fragments to neighboring buckets, finalization must propagate
carry across all limb buckets and write canonical 64-bit limbs.

## Fusion Points

There are two final output hooks:

- Pow2 path: replace the final `inv_range(..., len == n)` store with direct
  codec emission.
- PFA path: replace the store inside the final `inv_pfa_butterfly` natural-tile
  reinterpretation with direct codec emission. The store address already maps
  to a natural tile index.

## Bit Extraction Research

Surveyed implementations:

- `fast-pack/simdcomp` uses generated AVX/SSE unpack kernels for each bit
  width. The generated code loads vectors with unaligned loads, combines fixed
  shifts and masks, and dispatches through per-width function tables. This is a
  strong signal that peak throughput comes from compile-time-known widths and
  block-oriented unpacking.
- `fast-pack/LittleIntPacker` uses generated scalar functions inspired by
  TurboPFor. It consumes complete 64-bit words and fully unpacks fixed-size
  blocks of 32 integers, with boundary crossings handled by generated
  shift/or/mask code.
- `quickwit-oss/bitpacking` uses Rust macros and const bit widths to generate
  unrolled pack/unpack functions. Runtime `num_bits` is handled by dispatching
  to the generated fixed-width implementation rather than keeping the inner loop
  width-generic.

Links:

- https://github.com/fast-pack/simdcomp
- https://github.com/fast-pack/simdcomp/blob/master/scripts/avxpacking.py
- https://github.com/fast-pack/simdcomp/blob/master/src/avxbitpacking.c
- https://github.com/fast-pack/LittleIntPacker
- https://github.com/fast-pack/LittleIntPacker/blob/master/src/turbobitpacking32.c
- https://github.com/quickwit-oss/bitpacking
- https://github.com/quickwit-oss/bitpacking/blob/master/src/macros.rs

Consequences for this project:

- The proposed grouped runtime extraction remains the right scalar baseline.
  Loading a `group_bits <= 60` packet and splitting it into two complex pairs
  avoids four independent trunk extractions and reduces boundary handling to two
  bitfield loads per FFT input tile.
- Full block unpacking into a temporary integer array is probably a poor match.
  The FFT load wants four complex lanes in natural tile order, and PFA can use
  rotated/non-contiguous logical indices. Materializing a conventional unpacked
  block would add staging work that the FFT immediately repacks into vectors.
- SIMD bit unpacking in existing libraries works best when many consecutive
  integers with the same compile-time width are unpacked into an integer array.
  This differs from the desired codec shape, where the output is directly a
  `cv` tile and output emission may scatter-add into unaligned limb buckets.
- BMI2 `pext` is unlikely to be the first winning path for contiguous bitfields.
  The usual fast code for contiguous fields is still two word loads plus
  shift/or/mask. `pext` may be worth a targeted benchmark only if a future
  codec extracts non-contiguous lane layouts.
- AVX2 scatter is not available, and runtime output targets can share limb
  buckets. Output fusion should first use scalar scatter-add after vector
  rounding/extraction. A later AVX-512 path could be considered separately, but
  it should not shape the AVX2 baseline.

Recommended implementation order:

1. Keep `U16Codec` fully specialized and equivalent to the current input/output
   code.
2. Add `RuntimeBitsCodec` with grouped scalar input extraction and scalar
   bucketed output emission.
3. Add optional fixed-width runtime fast paths for hot widths such as 15, 14,
   and 13. These can use the same codec concept but dispatch to
   `RuntimeBitsCodecFixed<B>` internally so the compiler sees constant shifts
   and masks.
4. Benchmark before attempting SIMD trunk scattering. Existing libraries suggest
   generated fixed-width scalar code is a better next step than a generic SIMD
   scatter design.

## Implementation Status

- `src/fft.cpp` now has `U16Codec` and `RuntimeBitsCodec` wired through forward
  unpack, PFA unpack, fused pow2 inverse output, and fused PFA inverse output.
- `RuntimeBitsCodec` uses two grouped bitfield loads per four complex lanes and
  emits two sequential `u128` group partials per tile. The final
  `partial_to_limbs` pass handles runtime bit offsets and 64-bit limb packing.
- Runtime input decode has an eight-line encoded-address cache. Pow2 and PFA
  forward unpack both pass stable stream tags in the low three bits. Cache
  entries are now AoSoV `u32[8]` tiles and are refilled with sequential stream
  loaders.
- `src/fft.hpp` exposes experimental `mul_bits` and `sqr_bits` entry points.
  `trunk_bits == 16` dispatches to the existing specialized `mul`/`sqr` path.
- `src/fft.hpp` also exposes `mul_auto` and `sqr_auto`, which select trunk
  width from the transform-length bands below.
- The generated root `fft.hpp` has not been regenerated in this pass.

## Delegate Bands

For balanced multiplication, a `b`-bit trunk maps roughly
`N ~= ceil(64 * limbs / b)`. The delegate chooses the first trunk width whose
selected FFT length fits the band:

| transform length `N` | trunk bits | balanced limb limit |
| --- | ---: | ---: |
| `8..2^16` | 16 | `2^14 = 16,384` |
| `2^16..2^19` | 15 | `2^19 * 15 / 64 = 122,880` |
| `2^19..2^21` | 14 | `2^21 * 14 / 64 = 458,752` |
| `2^21..2^23` | 13 | `2^23 * 13 / 64 = 1,703,936` |

The 15-bit range intentionally keeps a one-bit margin because output packing
only shifts by 15 bits. The 14-bit and 13-bit ranges are bounded by the
available integer precision in double coefficients.

## Delegate Sweep

Benchmark artifacts:

- `bench/bench_trunk_delegate.cpp`
- `bench/plot_trunk_delegate.py`
- `bench/trunk_delegate.csv`
- `bench/trunk_delegate.png`
- `bench/bench_runtime_io.cpp`

The recorded sweep uses transform-length targets spaced at `2^(1/8)` and runs
through actual transform length `2^23`. GMP speedup is measured through 65,536
limbs; above that, the CSV leaves GMP columns empty and records FFT throughput
only.

`bench/bench_runtime_io.cpp` is a white-box benchmark for the runtime input
loader. It includes `src/fft.cpp` directly so it can validate internal codec
state without exposing benchmark-only APIs. The benchmark compares:

- independent grouped bitfield extraction,
- generic sequential stream refill,
- the actual codec refill path,
- cached codec tile load throughput.

Its startup validation compares independent refill, stream refill, codec refill,
and codec tile loads for all runtime widths `1..15`. It validates both truncated
tail segments and padded full segments, so both scalar fallback and SIMD refill
paths are covered.

Representative isolated result for 1,048,576 pairs on the current WSL/Clang
setup:

| trunk bits | independent refill ns/pair | codec refill ns/pair | speedup |
| ---: | ---: | ---: | ---: |
| 15 | 1.131637 | 0.124049 | 9.12x |
| 14 | 0.926017 | 0.125700 | 7.37x |
| 13 | 0.932373 | 0.124253 | 7.50x |

## Open Questions

- Whether the fixed 13/14/15 stream refill can be improved further with a
  wider unroll or byte/tile-oriented loads.
- Whether runtime output scatter can be profitably vectorized with AVX2. Native
  scatter is not available, and the target addresses are bit-offset buckets, so
  scalar scatter may remain best.
- Whether generated specialized codecs for common runtime widths are worth the
  code size.

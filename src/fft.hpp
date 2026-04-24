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

// Experimental runtime digit width path. Inputs and outputs remain little-endian
// 64-bit limbs; trunk_bits selects the internal base-2^trunk_bits digit stream.
// trunk_bits must be in [1, 16]. The 16-bit case dispatches to mul/sqr.
int mul_bits(std::uint64_t* rp,
             const std::uint64_t* ap, std::ptrdiff_t an,
             const std::uint64_t* bp, std::ptrdiff_t bn,
             unsigned trunk_bits);

int sqr_bits(std::uint64_t* rp,
             const std::uint64_t* ap, std::ptrdiff_t an,
             unsigned trunk_bits);

// Delegate entry: choose 16/15/14/13-bit trunks from the selected transform
// length band, up to N = 2^23.
int mul_auto(std::uint64_t* rp,
             const std::uint64_t* ap, std::ptrdiff_t an,
             const std::uint64_t* bp, std::ptrdiff_t bn);

int sqr_auto(std::uint64_t* rp,
             const std::uint64_t* ap, std::ptrdiff_t an);

}  // namespace fft

/// Standalone benchmark comparing f32 vs int8 LUT residual scoring.
///
/// Compile and run:
///   rustc -O benches/score_residual.rs -o /tmp/bench_score && /tmp/bench_score
///
/// No external dependencies -- scoring kernels are inlined so the benchmark
/// can run without building the full crate (which needs libtorch).
///
/// Reference: https://chaochunhsu.github.io/blog/slow-half-of-plaid/

use std::time::Instant;

// ---- scoring kernels (copied from rust/search/decompressor.rs) ----

fn build_reversed_bit_map(nbits: u8) -> [u8; 256] {
    let mut reversed = [0u8; 256];
    let nbits_mask = (1 << nbits) - 1;
    for byte_val in 0..256u32 {
        let mut reversed_bits = 0u32;
        let mut bit_pos = 8;
        while bit_pos >= nbits {
            let segment = (byte_val >> (bit_pos - nbits)) & nbits_mask;
            let mut reversed_segment = 0u32;
            for k in 0..nbits {
                if (segment & (1 << k)) != 0 {
                    reversed_segment |= 1 << (nbits - 1 - k);
                }
            }
            reversed_bits |= reversed_segment;
            if bit_pos > nbits {
                reversed_bits <<= nbits;
            }
            bit_pos -= nbits;
        }
        reversed[byte_val as usize] = (reversed_bits & 0xFF) as u8;
    }
    reversed
}

fn decompress_residual_4bit(
    residual: &[u8],
    reversed_bit_map: &[u8; 256],
    bucket_scores: &[f32],
    bucket_dim_shift: usize,
) -> f32 {
    let mut score = 0.0f32;
    for (packed_idx, &packed_val) in residual.iter().enumerate() {
        let packed_val = reversed_bit_map[packed_val as usize];
        let unpacked_idx_0 = packed_idx << 1;
        let unpacked_idx_1 = unpacked_idx_0 + 1;
        let unpacked_0 = (packed_val >> 4) as usize;
        let unpacked_1 = (packed_val & 0x0F) as usize;
        let idx0 = (unpacked_idx_0 << bucket_dim_shift) | unpacked_0;
        let idx1 = (unpacked_idx_1 << bucket_dim_shift) | unpacked_1;
        score += bucket_scores[idx0] + bucket_scores[idx1];
    }
    score
}

fn decompress_residual_2bit(
    residual: &[u8],
    reversed_bit_map: &[u8; 256],
    bucket_scores: &[f32],
    bucket_dim_shift: usize,
) -> f32 {
    let mut score = 0.0f32;
    for (packed_idx, &packed_val) in residual.iter().enumerate() {
        let packed_val = reversed_bit_map[packed_val as usize];
        let d0 = packed_idx << 2;
        let c0 = (packed_val >> 6) as usize;
        let c1 = ((packed_val >> 4) & 0x03) as usize;
        let c2 = ((packed_val >> 2) & 0x03) as usize;
        let c3 = (packed_val & 0x03) as usize;
        score += bucket_scores[(d0 << bucket_dim_shift) | c0]
            + bucket_scores[((d0 + 1) << bucket_dim_shift) | c1]
            + bucket_scores[((d0 + 2) << bucket_dim_shift) | c2]
            + bucket_scores[((d0 + 3) << bucket_dim_shift) | c3];
    }
    score
}

fn build_int8_lut(
    bucket_scores: &[f32],
    num_tokens: usize,
    dim: usize,
    num_buckets: usize,
    nbits: u8,
) -> (Vec<i8>, Vec<f32>) {
    let stride = dim * num_buckets;
    let code_rev: Vec<u8> = (0..num_buckets)
        .map(|code| {
            let mut r = 0u8;
            for bit in 0..nbits {
                if (code as u8) & (1 << bit) != 0 {
                    r |= 1 << (nbits - 1 - bit);
                }
            }
            r
        })
        .collect();

    let mut weights = vec![0i8; num_tokens * stride];
    let mut scales = vec![0.0f32; num_tokens];

    for token in 0..num_tokens {
        let offset = token * stride;
        let token_scores = &bucket_scores[offset..offset + stride];
        let abs_max = token_scores.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max > 1e-10 { abs_max / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        scales[token] = scale;

        for d in 0..dim {
            for raw_code in 0..num_buckets {
                let reversed_code = code_rev[raw_code] as usize;
                let f32_val = token_scores[d * num_buckets + reversed_code];
                let quantized = (f32_val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                weights[offset + d * num_buckets + raw_code] = quantized;
            }
        }
    }
    (weights, scales)
}

#[inline]
fn score_residual_4bit_int8(residual: &[u8], lut: &[i8]) -> i32 {
    let mut sum: i32 = 0;
    for (i, &packed) in residual.iter().enumerate() {
        let d0 = i << 1;
        let d1 = d0 + 1;
        let hi = (packed >> 4) as usize;
        let lo = (packed & 0x0F) as usize;
        sum += lut[(d0 << 4) | hi] as i32;
        sum += lut[(d1 << 4) | lo] as i32;
    }
    sum
}

#[inline]
fn score_residual_2bit_int8(residual: &[u8], lut: &[i8]) -> i32 {
    let mut sum: i32 = 0;
    for (i, &packed) in residual.iter().enumerate() {
        let d0 = i << 2;
        let c0 = (packed >> 6) as usize;
        let c1 = ((packed >> 4) & 0x03) as usize;
        let c2 = ((packed >> 2) & 0x03) as usize;
        let c3 = (packed & 0x03) as usize;
        sum += lut[(d0 << 2) | c0] as i32;
        sum += lut[((d0 + 1) << 2) | c1] as i32;
        sum += lut[((d0 + 2) << 2) | c2] as i32;
        sum += lut[((d0 + 3) << 2) | c3] as i32;
    }
    sum
}

// ---- benchmark harness ----

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

fn bench_batch<F: Fn() -> f32>(label: &str, num_embeddings: usize, f: F) -> f64 {
    // warmup
    for _ in 0..WARMUP_ITERS {
        std::hint::black_box(f());
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        std::hint::black_box(f());
        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ns = times[BENCH_ITERS / 2];
    let ns_per_emb = median_ns / num_embeddings as f64;
    eprintln!("  {label:>12}: {ns_per_emb:6.1} ns/embedding  (median of {BENCH_ITERS} runs)");
    ns_per_emb
}

fn run_4bit_bench() {
    let dim = 128;
    let num_buckets = 16usize;
    let num_tokens = 32;
    let num_embeddings = 8192;
    let stride = dim * num_buckets;
    let residual_bytes = dim / 2;

    eprintln!("\n=== 4-bit residual scoring ===");
    eprintln!("  dim={dim}, tokens={num_tokens}, embeddings={num_embeddings}");
    eprintln!("  f32 LUT size per token: {} bytes", stride * 4);
    eprintln!("  i8  LUT size per token: {} bytes", stride);

    let bucket_scores: Vec<f32> = (0..num_tokens * stride)
        .map(|i| (i as f32 * 0.7123).sin() * 0.5)
        .collect();
    let reversed_bit_map = build_reversed_bit_map(4);
    let (i8_lut, scales) = build_int8_lut(&bucket_scores, num_tokens, dim, num_buckets, 4);

    let residuals: Vec<Vec<u8>> = (0..num_embeddings)
        .map(|e| {
            (0..residual_bytes)
                .map(|i| ((e * 37 + i * 13 + 7) & 0xFF) as u8)
                .collect()
        })
        .collect();

    let f32_ns = bench_batch("f32 LUT", num_embeddings, || {
        let mut total = 0.0f32;
        for (idx, res) in residuals.iter().enumerate() {
            let token = idx % num_tokens;
            let off = token * stride;
            total += decompress_residual_4bit(
                res,
                &reversed_bit_map,
                &bucket_scores[off..off + stride],
                4,
            );
        }
        total
    });

    let i8_ns = bench_batch("int8 LUT", num_embeddings, || {
        let mut total = 0.0f32;
        for (idx, res) in residuals.iter().enumerate() {
            let token = idx % num_tokens;
            let off = token * stride;
            let raw = score_residual_4bit_int8(res, &i8_lut[off..off + stride]);
            total += raw as f32 * scales[token];
        }
        total
    });

    eprintln!("  speedup: {:.2}x", f32_ns / i8_ns);
}

fn run_2bit_bench() {
    let dim = 128;
    let num_buckets = 4usize;
    let num_tokens = 32;
    let num_embeddings = 8192;
    let stride = dim * num_buckets;
    let residual_bytes = dim / 4;

    eprintln!("\n=== 2-bit residual scoring ===");
    eprintln!("  dim={dim}, tokens={num_tokens}, embeddings={num_embeddings}");
    eprintln!("  f32 LUT size per token: {} bytes", stride * 4);
    eprintln!("  i8  LUT size per token: {} bytes", stride);

    let bucket_scores: Vec<f32> = (0..num_tokens * stride)
        .map(|i| (i as f32 * 0.3917).cos() * 0.4)
        .collect();
    let reversed_bit_map = build_reversed_bit_map(2);
    let (i8_lut, scales) = build_int8_lut(&bucket_scores, num_tokens, dim, num_buckets, 2);

    let residuals: Vec<Vec<u8>> = (0..num_embeddings)
        .map(|e| {
            (0..residual_bytes)
                .map(|i| ((e * 41 + i * 17 + 3) & 0xFF) as u8)
                .collect()
        })
        .collect();

    let f32_ns = bench_batch("f32 LUT", num_embeddings, || {
        let mut total = 0.0f32;
        for (idx, res) in residuals.iter().enumerate() {
            let token = idx % num_tokens;
            let off = token * stride;
            total += decompress_residual_2bit(
                res,
                &reversed_bit_map,
                &bucket_scores[off..off + stride],
                2,
            );
        }
        total
    });

    let i8_ns = bench_batch("int8 LUT", num_embeddings, || {
        let mut total = 0.0f32;
        for (idx, res) in residuals.iter().enumerate() {
            let token = idx % num_tokens;
            let off = token * stride;
            let raw = score_residual_2bit_int8(res, &i8_lut[off..off + stride]);
            total += raw as f32 * scales[token];
        }
        total
    });

    eprintln!("  speedup: {:.2}x", f32_ns / i8_ns);
}

fn main() {
    eprintln!("Int8 LUT scoring benchmark");
    eprintln!("Reference: https://chaochunhsu.github.io/blog/slow-half-of-plaid/");
    run_4bit_bench();
    run_2bit_bench();
    eprintln!();
}

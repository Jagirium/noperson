//! Affine transform estimation.
//!
//! Replaces:
//! - `skimage.transform.SimilarityTransform` → `similarity_transform()`
//! - `cv2.estimateAffinePartial2D(method=LMEDS)` → `estimate_affine_lmeds()`
//!
//! These operate on 5 points (10 scalars) — too small for GPU kernel launch overhead.
//! The resulting 2×3 matrix is immediately uploaded to GPU for NPP warp.

/// NPP-compatible affine coefficients: `[[a00, a01, a02], [a10, a11, a12]]`.
pub type Affine2x3 = [[f64; 3]; 2];

/// Least-squares 2D similarity alignment from src → dst.
///
/// Returns the six affine coefficients directly in NPP's 2×3 layout.
///
/// The closed form solves the same four-parameter model as
/// `skimage.transform.SimilarityTransform.estimate`:
/// `u = a*x - b*y + tx`, `v = b*x + a*y + ty`.
/// Keeping this exact model matters because Inswapper is sensitive to even a
/// small crop-scale mismatch.
pub fn similarity_transform(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Affine2x3 {
    let n = src.len();
    assert!(n >= 2, "need at least 2 point correspondences");
    assert_eq!(n, dst.len());

    // Compute centroids
    let (mut src_cx, mut src_cy) = (0.0f64, 0.0f64);
    let (mut dst_cx, mut dst_cy) = (0.0f64, 0.0f64);
    for i in 0..n {
        src_cx += src[i][0] as f64;
        src_cy += src[i][1] as f64;
        dst_cx += dst[i][0] as f64;
        dst_cy += dst[i][1] as f64;
    }
    let nf = n as f64;
    src_cx /= nf;
    src_cy /= nf;
    dst_cx /= nf;
    dst_cy /= nf;

    // Solve the centered least-squares system for rotation*scale.
    let mut denominator = 0.0f64;
    let mut a_numerator = 0.0f64;
    let mut b_numerator = 0.0f64;
    for i in 0..n {
        let sx = src[i][0] as f64 - src_cx;
        let sy = src[i][1] as f64 - src_cy;
        let dx = dst[i][0] as f64 - dst_cx;
        let dy = dst[i][1] as f64 - dst_cy;
        denominator += sx * sx + sy * sy;
        a_numerator += sx * dx + sy * dy;
        b_numerator += sx * dy - sy * dx;
    }
    let (a, b) = if denominator.abs() < f64::EPSILON {
        (1.0, 0.0)
    } else {
        (a_numerator / denominator, b_numerator / denominator)
    };
    let tx = dst_cx - a * src_cx + b * src_cy;
    let ty = dst_cy - b * src_cx - a * src_cy;

    [[a, -b, tx], [b, a, ty]]
}

/// Invert a 2×3 affine matrix.
/// Returns the inverse 2×3 matrix such that applying it undoes the original transform.
pub fn invert_2x3(m: &[[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];

    let det = a * e - b * d;
    let inv_det = 1.0 / det;

    [
        [e * inv_det, -b * inv_det, (b * f - e * c) * inv_det],
        [-d * inv_det, a * inv_det, (d * c - a * f) * inv_det],
    ]
}

/// Compute affine matrix from 5-point face landmarks to canonical template.
///
/// This is the primary entry point for face alignment in the swap pipeline.
pub fn estimate_face_affine(
    landmarks_5: &[[f32; 2]; 5],
    template: &[[f32; 2]; 5],
) -> [[f64; 3]; 2] {
    similarity_transform(landmarks_5, template)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_affine_approximately_eq(actual: Affine2x3, expected: Affine2x3) {
        for row in 0..2 {
            for column in 0..3 {
                assert!(
                    (actual[row][column] - expected[row][column]).abs() < 1e-12,
                    "coefficient [{row}][{column}] = {} != {}",
                    actual[row][column],
                    expected[row][column]
                );
            }
        }
    }

    #[test]
    fn similarity_transform_returns_direct_translation_coefficients() {
        let src = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let dst = [[5.0, -3.0], [7.0, -3.0], [5.0, -1.0]];

        assert_affine_approximately_eq(
            similarity_transform(&src, &dst),
            [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]],
        );
    }

    #[test]
    fn similarity_transform_returns_uniform_scale_and_rotation_coefficients() {
        let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dst = [[1.0, -4.0], [1.0, -2.0], [-1.0, -4.0]];

        assert_affine_approximately_eq(
            similarity_transform(&src, &dst),
            [[0.0, -2.0, 1.0], [2.0, 0.0, -4.0]],
        );
    }

    #[test]
    fn similarity_transform_singular_input_uses_identity_linear_fallback() {
        let src = [[2.0, 3.0], [2.0, 3.0]];
        let dst = [[9.0, 11.0], [10.0, 12.0]];

        assert_affine_approximately_eq(
            similarity_transform(&src, &dst),
            [[1.0, 0.0, 7.5], [0.0, 1.0, 8.5]],
        );
    }

    #[test]
    fn invert_2x3_composes_to_all_six_identity_coefficients() {
        let m = [[2.0, 0.5, 10.0], [0.3, 1.5, 20.0]];
        let inv = invert_2x3(&m);
        let composed = [
            [
                m[0][0] * inv[0][0] + m[0][1] * inv[1][0],
                m[0][0] * inv[0][1] + m[0][1] * inv[1][1],
                m[0][0] * inv[0][2] + m[0][1] * inv[1][2] + m[0][2],
            ],
            [
                m[1][0] * inv[0][0] + m[1][1] * inv[1][0],
                m[1][0] * inv[0][1] + m[1][1] * inv[1][1],
                m[1][0] * inv[0][2] + m[1][1] * inv[1][2] + m[1][2],
            ],
        ];
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        for row in 0..2 {
            for column in 0..3 {
                assert!(
                    (composed[row][column] - expected[row][column]).abs() < 1e-10,
                    "coefficient [{row}][{column}] = {} != {}",
                    composed[row][column],
                    expected[row][column]
                );
            }
        }
    }

    #[test]
    fn similarity_transform_matches_crosswap_skimage_regression() {
        let landmarks = [
            [1307.7467, 609.63947],
            [1560.6035, 601.45264],
            [1407.266, 752.91705],
            [1342.9265, 827.84937],
            [1563.0288, 819.9446],
        ];
        let factor = 4.0;
        let mut template = crate::math::constants::ARCFACE_DST;
        for point in &mut template {
            point[0] *= factor;
            point[1] *= factor;
        }
        template[0][0] += factor * 8.0;
        template[0][1] += factor * 8.0;

        let actual = estimate_face_affine(&landmarks, &template);
        let expected = [
            [0.549040042, -0.030493923, -536.061834],
            [0.030493923, 0.549040042, -146.400637],
        ];

        for row in 0..2 {
            for column in 0..3 {
                assert!(
                    (actual[row][column] - expected[row][column]).abs() < 1e-4,
                    "matrix[{row}][{column}]={} != {}",
                    actual[row][column],
                    expected[row][column]
                );
            }
        }
    }
}

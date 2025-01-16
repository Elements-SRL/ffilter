use std::cmp::Ordering;
use ndarray::Array2;
use num_complex::Complex;

#[derive(PartialEq)]
pub enum FilterType {
    Lowpass,
    Highpass,
}

fn biquads(zz: &[Complex<f64>], zp: &[Complex<f64>], filter_type: FilterType) -> (Array2<f64>, Array2<f64>) {
    let n = zz.len();
    let nf = n / 2;
    let nc = (n + 1) / 2; // Equivalent to ceil(n/2)
    let mut b = Array2::<f64>::ones((nc, 3));
    let mut a = Array2::<f64>::ones((nc, 3));

    let lhs = match filter_type {
        FilterType::Lowpass => 1.0,
        FilterType::Highpass => -1.0,
    };

    // Process pairs of poles and zeros
    for idx in 0..nf {
        let zp_real = zp[idx].re;
        let zz_real = zz[idx].re;

        a[[idx, 1]] = -2.0 * zp_real;
        a[[idx, 2]] = zp[idx].norm_sqr();
        b[[idx, 1]] = -2.0 * zz_real;
        b[[idx, 2]] = zz[idx].norm_sqr();

        let g = (a[[idx, 0]] + lhs * a[[idx, 1]] + a[[idx, 2]])
            / (b[[idx, 0]] + lhs * b[[idx, 1]] + b[[idx, 2]]);
        b.row_mut(idx).mapv_inplace(|val| val * g);
    }
    // Handle the remaining pole/zero if n is odd
    if nc > nf {
        let zp_real = zp[nc - 1].re;
        let zz_real = zz[nc - 1].re;

        a[[nc - 1, 1]] = -zp_real;
        a[[nc - 1, 2]] = 0.0;
        b[[nc - 1, 1]] = -zz_real;
        b[[nc - 1, 2]] = 0.0;

        let g = (a[[nc - 1, 0]] + lhs * a[[nc - 1, 1]])
            / (b[[nc - 1, 0]] + lhs * b[[nc - 1, 1]]);
        b.row_mut(nc - 1).mapv_inplace(|val| val * g);
    }

    (b, a)
}


/// Compute biquad coefficients for a Butterworth filter.
///
/// # Arguments
/// - `n`: Order of the filter.
/// - `sr`: Sampling rate.
/// - `bw`: Bandwidth (cutoff frequency).
/// - `filter_type`: Filter type ('l' for low-pass, 'h' for high-pass).
///
/// # Returns
/// - `(b, a)`: Tuple containing the numerator (`b`) and denominator (`a`) coefficients.
pub fn butter_biquads(
    n: usize,
    sr: f64,
    bw: f64,
    filter_type: Option<FilterType>,
) -> (Array2<f64>, Array2<f64>) {
    let filter_type = filter_type.unwrap_or(FilterType::Lowpass); // Default to 'l' if not provided.

    // Step 1: Compute analog poles and zeros for a Butterworth filter.
    let (z, p) = butter_zp(n, sr, bw);

    // Step 2: If high-pass, adjust poles and zeros.
    let (z, p) = if filter_type == FilterType::Highpass {
        high_pass(&z, &p, sr, bw)
    } else {
        (z, p)
    };

    // Step 3: Convert analog poles and zeros to digital.
    let (zz, zp) = dig_zp(&z, &p);

    // Step 4: Compute biquad coefficients.
    biquads(&zz, &zp, filter_type)
}

/// Compute Butterworth poles.
///
/// # Arguments
/// - `n`: Order of the filter.
/// - `sr`: Sampling rate.
/// - `bw`: Bandwidth (cutoff frequency).
///
/// # Returns
/// - `(z, p)`: Tuple containing zeros (`z`) and poles (`p`) as vectors of complex numbers.
fn butter_zp(n: usize, sr: f64, bw: f64) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let w = freq_warp(bw, sr);

    // Zeros are empty for a Butterworth filter.
    let z: Vec<Complex<f64>> = Vec::new();

    // Compute poles in the s-plane.
    let mut p = Vec::new();
    for k in 0..n {
        let num = n as f64 + (2 * k + 1) as f64;
        let angle = num * std::f64::consts::PI / (2.0 * n as f64);
        p.push(w * Complex::new(angle.cos(), angle.sin()));
    }

    // Replace nearly real poles (imaginary part close to 0) with real values.
    for pole in &mut p {
        if pole.im.abs() < 1e-10 {
            *pole = Complex::new(pole.re, 0.0);
        }
    }

    (z, p)
}

/// Frequency warping function.
///
/// # Arguments
/// - `f`: Frequency in Hz.
/// - `sr`: Sampling rate in Hz.
///
/// # Returns
/// - Warped frequency as `f64`.
fn freq_warp(f: f64, sr: f64) -> f64 {
    2.0 * (std::f64::consts::PI * f / sr).tan()
}

fn high_pass(
    z: &[Complex<f64>],
    p: &[Complex<f64>],
    sr: f64,
    bw: f64,
) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let w = freq_warp(bw, sr);
    let r = w * w;

    // Reflect zeros and poles around the warped cutoff frequency squared.
    let mut new_z: Vec<Complex<f64>> = z.iter().map(|zero| r / zero).collect();
    let new_p: Vec<Complex<f64>> = p.iter().map(|pole| r / pole).collect();

    // Ensure zeros match the number of poles by padding with zeros.
    if new_p.len() > new_z.len() {
        new_z.resize(new_p.len(), Complex::new(0.0, 0.0));
    }

    (new_z, new_p)
}

/// Convert analog poles and zeros to digital poles and zeros.
///
/// # Arguments
/// - `z`: Vector of zeros (complex numbers).
/// - `p`: Vector of poles (complex numbers).
///
/// # Returns
/// - `(zz, zp)`: Digital zeros and poles as vectors of complex numbers.
fn dig_zp(
    z: &[Complex<f64>],
    p: &[Complex<f64>],
) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    // Transform analog zeros and poles to digital equivalents.
    let mut zz: Vec<Complex<f64>> = z.iter().map(|s| tform(*s)).collect();
    let mut zp: Vec<Complex<f64>> = p.iter().map(|s| tform(*s)).collect();

    // Ensure equal length by padding with -1.
    match zz.len().cmp(&zp.len()) {
        Ordering::Greater => zp.resize(zz.len(), Complex::new(-1.0, 0.0)),
        Ordering::Less => zz.resize(zp.len(), Complex::new(-1.0, 0.0)),
        Ordering::Equal => (),
    }
    (zz, zp)
}

/// Perform the bilinear transform on a complex value.
///
/// # Arguments
/// - `s`: A value in the analog domain (complex number).
///
/// # Returns
/// - The corresponding value in the digital domain (complex number).
fn tform(s: Complex<f64>) -> Complex<f64> {
    (Complex::new(2.0, 0.0) + s) / (Complex::new(2.0, 0.0) - s)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let (b, a) = butter_biquads(5, 5000.0, 2000.0, None);
        println!("{:?}, {:?}", b, a);
    }
}

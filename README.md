# Signal Processing Library

This library provides functions for calculating biquad coefficients of Butterworth filters, including lowpass and highpass filters. It uses analog-to-digital transformation techniques like the bilinear transform and frequency warping.

## Features

- **Butterworth Filter Design**: Calculates biquad coefficients for Butterworth filters.
- **Lowpass & Highpass Filters**: Supports both filter types.
- **Complex Number Support**: Uses `num_complex` for complex operations.
- **Analog to Digital Transformation**: Converts analog filter specs to digital coefficients.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.16"
num-complex = "0.4"

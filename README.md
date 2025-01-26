# moment-matching-calibration
Implementation of the Moment-Matching Market Implied Calibration Method (F. Guillaume and W.Schoutens, 2012).
The current implementation is for some popular infinte activity Levy models: VG (Meixner and NIG still to be implemented).
The methodology is benchmarked against standard calibration methods.

Pricing of vanilla options is done via the Carr-Madan FFT. This calculation is flexible to adapt to any characteristic function.


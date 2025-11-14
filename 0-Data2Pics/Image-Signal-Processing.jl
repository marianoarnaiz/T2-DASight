# ================================================================
#  DASight Functions: Image/Signal processing
#  Author: Mariano Arnaiz
#  Description:
#    This file contains several functions used for process DAS.
#    All the functions here are used to train a ML ResNet or to process
#    it as input for classification
# ================================================================


# ---- 0.1 Modules ----------------------------------------------------
using FFTW, DSP, Statistics
using ImageFiltering: Kernel, imfilter
using FileIO, Images, Colors, Statistics, ColorVectorSpace
using Dates
using NCDatasets, DASVader, CairoMakie, NanoDates, Dates, StatsBase

# ---- 1. Get Band Modules ----------------------------------------------------
#   Functions here aid in automatically band-pass filtering the data. We search for dominant low freq!

# --- Welch-like PSD per trace: split into windows, windowed FFT, average ---

"""
    welch_psd_trace(x::AbstractVector, dt; seglen=4096, overlap=0.5, window=hanning)

Compute a Welch-style Power Spectral Density (PSD) estimate for a single DAS trace.

This function:
- Splits the trace into overlapping segments.
- Applies a window function to each segment (default: `hanning`).
- Computes the one-sided FFT using `rfft`.
- Averages the power spectra across all segments.
- Normalizes by window power and sampling frequency.

# Arguments
- `x`: 1D signal vector (one DAS channel in time).
- `dt`: Sampling interval (seconds).
- `seglen`: Length of each Welch segment (default 4096 samples).
- `overlap`: Fractional segment overlap (0–1). Default 0.5.
- `window`: Window function applied per segment (default: `hanning`).

# Returns
- `freqs`: Frequency axis for the one-sided PSD.
- `P`: Averaged PSD for this trace.

This implementation is similar to MATLAB’s `pwelch` or SciPy’s `signal.welch`.
"""
function welch_psd_trace(x::AbstractVector, dt; seglen=4096, overlap=0.5, window=hanning)
    n = length(x)
    step = Int(seglen * (1 - overlap))             # step between segments
    if step < 1
        error("increase seglen or reduce overlap")
    end

    starts = 1:step:(n-seglen+1)                   # starting indices of segments
    m = length(starts)                             # number of segments
    fs = 1/dt                                      # sampling frequency

    # One-sided freq axis for rfft
    freqs = collect(0:fs/seglen:fs/2)

    Pacc = zeros(length(freqs))                    # accumulator for mean PSD

    w = window(seglen)                             # analysis window
    U = sum(abs2, w)                               # window power normalization

    for s in starts
        seg = x[s:(s+seglen-1)] .* w               # apply window
        X = rfft(seg)                              # FFT of segment
        # accumulate PSD (only positive freqs)
        Pacc .+= abs.(X[1:length(freqs)]).^2 ./ (U * fs)
    end

    return freqs, Pacc ./ m                        # average across segments
end


# --- build median PSD across windows & traces ---

"""
    median_psd_welch(dDAS; seglen=4096, overlap=0.5)

Compute a robust PSD estimate across *all* traces (channels) of a DAS dataset.

Steps:
1. Apply `welch_psd_trace` to each trace.
2. Stack all PSDs into a matrix (traces × frequencies).
3. Compute:
   - Median PSD across traces (robust to noisy channels).
   - A lower percentile PSD (15th percentile) as an estimate of noise floor.

# Arguments
- `dDAS`: A DAS data structure containing `data` and `time` fields.
- `seglen`: Welch segment length.
- `overlap`: Segment overlap fraction.

# Returns
- `freqs`: Frequency axis.
- `median_psd`: Median PSD across all traces.
- `noise_psd`: Noise-floor estimate from 15% quantile.
- `Pmat`: The PSD matrix (ntraces × nfreq).
"""
function median_psd_welch(dDAS; seglen=4096, overlap=0.5)
    dt = dDAS.time[2]-dDAS.time[1]
    ntraces = size(dDAS.data, 2)

    freqs = nothing
    PSDs = []

    # Compute Welch PSD for each trace
    for j in 1:ntraces
        f, P = welch_psd_trace(dDAS.data[:, j], dt; seglen=seglen, overlap=overlap)
        freqs === nothing && (freqs = f)           # set frequency axis only once
        push!(PSDs, P)
    end

    # Convert vector-of-vectors to matrix (traces × frequencies)
    Pmat = hcat(PSDs...)'

    # Robust median across traces
    median_psd = mapslices(median, Pmat; dims=1)[:]

    # Estimate noise floor (lower percentile)
    noise_psd = mapslices(x->quantile(x, 0.15), Pmat; dims=1)[:]

    return freqs, median_psd, noise_psd, Pmat
end


# --- smooth in dB by moving median or gaussian kernel ---

"""
    smooth_db(vec; halfwidth_bins=5)

Apply median smoothing to a PSD curve (in dB).

- Converts PSD to dB.
- Applies a running median filter with window ± `halfwidth_bins`.

Useful for reducing small-scale fluctuations in PSD/SNR curves.

# Arguments
- `vec`: Input PSD vector (linear scale).
- `halfwidth_bins`: Number of bins on each side for smoothing.

# Returns
- Smoothed PSD curve in dB.
"""
function smooth_db(vec; halfwidth_bins=5)
    logp = 10 .* log10.(vec .+ eps(Float64))       # avoid log(0)
    n = length(logp)
    out = similar(logp)
    k = halfwidth_bins

    # Running median
    for i in 1:n
        lo = max(1, i-k)
        hi = min(n, i+k)
        out[i] = median(logp[lo:hi])
    end
    return out
end


"""
    bump_band(freqs, SNR_med; mindrop_db=2.0, smooth_bins=10)

Identify a dominant frequency "bump" (band) in the median SNR spectrum.

Procedure:
1. Optionally smooth the SNR curve to remove wiggles.
2. Find the peak frequency.
3. Search left/right until the SNR drops by ≥ `mindrop_db` dB.
4. Return a frequency band around the peak.

Used to define an automatic bandpass filter.

# Arguments
- `freqs`: Frequency axis (1D vector).
- `SNR_med`: Median SNR curve.
- `mindrop_db`: Drop from peak that defines band edges.
- `smooth_bins`: Smoothing window for the SNR curve.

# Returns
`(fmin, fcenter, fmax)`
A tuple with lower bound, peak frequency, and upper bound (rounded).
"""
function bump_band(freqs, SNR_med; mindrop_db=2.0, smooth_bins=10)
    N = length(SNR_med)
    half = div(N, 2)                                # use only positive frequencies
    f = freqs[1:half]
    S = SNR_med[1:half]

    # Smooth SNR curve
    S_smooth = similar(S)
    for i in 1:length(S)
        lo = max(1, i-smooth_bins)
        hi = min(length(S), i+smooth_bins)
        S_smooth[i] = median(S[lo:hi])
    end

    # Peak location
    peak_idx = argmax(S_smooth)
    peak_freq = f[peak_idx]
    peak_val = S_smooth[peak_idx]

    # Left boundary
    left_idx = 1
    for i in peak_idx:-1:2
        if peak_val - S_smooth[i] >= mindrop_db
            left_idx = i
            break
        end
    end

    # Right boundary
    right_idx = half
    for i in peak_idx:half-1
        if peak_val - S_smooth[i] >= mindrop_db
            right_idx = i
            break
        end
    end

    return (floor(f[left_idx]), round(f[peak_idx]), ceil(f[right_idx]))
end


"""
    get_dp(dDAS; seglen=4096)

Compute the optimal frequency band for DAS preprocessing:
- Uses Welch PSD across all traces.
- Computes median PSD and noise PSD.
- Computes median SNR = medianPSD - noisePSD.
- Smooths both curves.
- Extracts bandpass limits using `bump_band`.

# Arguments
- `dDAS`: DAS data object (must contain `data` and `time`).
- `seglen`: Welch segment length.

# Returns
- `dbmin`: Lower frequency bound.
- `dbmax`: Upper frequency bound.
- `freqs`: Frequency axis.
- `SNR_med`: Median SNR curve.
"""
function get_dp(dDAS; seglen = 4096)
    # Compute PSD across traces
    freqs, median_psd, noise_psd, Pmat = median_psd_welch(dDAS; seglen=seglen, overlap=0.5)

    # Smooth in dB
    sm_med = smooth_db(median_psd; halfwidth_bins=floor(Int,size(median_psd,1)*(0.5/100)))
    noise_med = smooth_db(noise_psd; halfwidth_bins=floor(Int,size(noise_psd,1)*(10/100)))

    # Median SNR
    SNR_med = sm_med .- noise_med

    # Extract frequency band around dominant bump
    dbmin, bdcentre, dbmax = bump_band(
        freqs[1:div(size(SNR_med,1),2)],
        SNR_med[1:div(size(SNR_med,1),2)];
        mindrop_db = round(mean(SNR_med[1:div(size(SNR_med,1),2)])),
        smooth_bins = floor(Int,size(SNR_med[1:div(size(SNR_med,1),2)],1)*(0.5/100))
    )

    return dbmin, dbmax, freqs, SNR_med
end


# ---- 2. FK smoothing ----------------------------------------------------
#   Functions here are for transforming the data in FK domian, finding the peak and smoothing with a 2D gaussian filter

"""
    fkpeak(zs, Δx, Δt; dbmin=0.0, dbmax=Inf, taper=0.05)

Compute the 2-D FK spectrum (folded, positive k and f only), mask
everything outside dbmin…dbmax (Hz), remove isolated hot pixels with
a 5×5 median filter, and return the (f, k) coordinates of the global
maximum.

Units
-----
zs      :: Matrix{<:Real}   size(zs) = (nt, nx)  →  (time, space)
Δx      :: Real             channel spacing [m]
Δt      :: Real             sample interval [s]
dbmin   :: Real             lower frequency cut-off [Hz]
dbmax   :: Real             upper frequency cut-off [Hz]
taper   :: Real             Tukey taper ratio (0–1)

Returns
-------
f_selected :: Float64       frequency of peak [Hz]
k_selected :: Float64       wavenumber of peak [rad/m]
f_pos      :: Vector        frequency axis (positive) [Hz]
k_pos      :: Vector        wavenumber axis (positive) [rad/m]
Pdb_smooth :: Matrix        cleaned spectrum (dB)
"""
function fkpeak(zs, Δx, Δt; dbmin=0.0, dbmax=Inf, taper=0.05)
    nt, nx = size(zs)

    # ----- 2. pre-processing -------------------------------------------------
    zs = copy(zs)                              # keep caller's array intact
    zs .-= mean(zs, dims=2)
    zs .-= mean(zs, dims=1)
    tap_t = DSP.Windows.tukey(nt, taper)
    tap_x = DSP.Windows.tukey(nx, taper)
    zs  .= zs .* tap_t .* tap_x'

    # ----- 3. 2-D FFT --------------------------------------------------------
    Δk = 2π / (nx * Δx)
    Δf = 1 / (nt * Δt)
    FFT = fftshift(fft(zs))

    # ----- 4. positive f and k only ------------------------------------------
    k_pos_idx = nx÷2+1 : nx
    f_pos_idx = nt÷2+1 : nt
    k_pos = (0 : length(k_pos_idx)-1) .* Δk
    f_pos = (0 : length(f_pos_idx)-1) .* Δf
    FFT_pos = FFT[f_pos_idx, k_pos_idx]          # (nf+, nk+)

    # ----- 5. fold ------------------------------------------------------------
    k_neg_idx = nx÷2 : -1 : 1
    FFT_fold  = similar(FFT_pos)
    for (j, (kp, kn)) in enumerate(zip(k_pos_idx, k_neg_idx))
        @views FFT_fold[:, j] = FFT_pos[:, j] .+ FFT[f_pos_idx, kn]
    end

    # ----- 6. power + mask ----------------------------------------------------
    Pdb = @. 10 * log10(abs2(FFT_fold) + eps(real(FFT_fold)))
    mask = @. (f_pos < dbmin) | (f_pos > dbmax)
    Pdb_masked = copy(Pdb)
    Pdb_masked[mask, :] .= 0.0

    # ----- 7. median filter ---------------------------------------------------
    function median5x5!(A)
        B = similar(A)
        R = CartesianIndices(A)
        Ifirst, Ilast = first(R), last(R)
        for I in R
            imin = max(Ifirst, I - CartesianIndex(2,2))
            imax = min(Ilast,  I + CartesianIndex(2,2))
            B[I] = median!(view(A, imin:imax))
        end
        B
    end
    Pdb_smooth = median5x5!(Pdb_masked)

    # ----- 8. peak pick -------------------------------------------------------
    imax = argmax(Pdb_smooth)
    f_selected = f_pos[imax[1]]
    k_selected = k_pos[imax[2]]

    return f_selected, k_selected, f_pos, k_pos, Pdb_smooth
end

######################################################################

"""
    fkgaussfilter(zs, Δx, Δt, f_pk, k_pk; relwidth=0.1)

Blur `zs` with a 2-D Gaussian whose widths are set from the FK peak
f_pk (Hz) and k_pk (rad/m).

`relwidth` gives the Gaussian σ as a fraction of the peak wavelength
in each direction (default 10 %).

Returns
-------
zs_blur :: Matrix{Float64}   same size as input
"""
function fkgaussfilter(zs, Δx, Δt, f_pk, k_pk; relwidth=0.1)
    # wavelength → samples
    λt = 1 / f_pk          # seconds
    λx = 2π / k_pk         # metres
    σt = relwidth * λt / Δt   # samples along time
    σx = relwidth * λx / Δx   # samples along space

    # kernel size (odd, ≥ 3)
    wt = max(3, ceil(Int, 4σt) ÷ 2 * 2 + 1)
    wx = max(3, ceil(Int, 4σx) ÷ 2 * 2 + 1)

    kernel = Kernel.gaussian((σt, σx), (wt, wx))
    imfilter(zs, kernel, "replicate")
end


# ---- 3. Automatic Level Intensity Adjustment --------------------------------------------------------
#   These functions automatically correct the colors of the images. They look more "vivid" and
#   helps make the larger values clearer. Inspired by auto-adjust feature or Mac OS's Preview!

"""
    auto_level_intensity(img; tail=0.005)

Apply automatic intensity leveling (contrast stretching) to an image.

This function:
- Converts the input image to RGB (preserving alpha channel if present).
- Computes luminance (`Gray`) to estimate brightness levels.
- Finds lower and upper intensity quantiles using `tail` (default 0.5%).
- Performs linear contrast stretching:
      new = clamp((x - lo) / (hi - lo), 0, 1)
- Applies the transformation independently to R, G, and B channels.
- Returns an RGB or RGBA image depending on the input.

This is equivalent to "auto-levels" or "contrast stretch" in image-processing
software, improving visibility of patterns without changing relative shapes.

# Arguments
- `img`: Input image (can be RGB, RGBA, Gray, etc.).
- `tail`: Fraction of pixels ignored at low/high ends when computing quantiles.
          Typical values: 0.001–0.02.

# Returns
- A contrast-enhanced image with the same type (RGB or RGBA) as input.
"""
function auto_level_intensity(img; tail=0.005)
    col = RGB.(img)                  # extract RGB channels
    α   = alpha.(img)                # store alpha channel (if present)
    col = float.(col)                # convert RGB to floating-point

    # Compute luminance (brightness) of each pixel
    lum = Gray.(col)

    # Flatten luminance to a vector and compute low/high quantiles
    lo = quantile(vec(channelview(lum)), tail)
    hi = quantile(vec(channelview(lum)), 1 - tail)
    Δ  = hi - lo                     # dynamic range

    # Contrast-stretching transformation
    f(x) = Δ > 0 ? clamp((x - lo) / Δ, 0.0, 1.0) : x

    # Apply transformation to each RGB channel
    out = map(c -> RGB(f(c.r), f(c.g), f(c.b)), col)

    # If original image had alpha, return RGBA; otherwise RGB
    return eltype(img) <: ColorAlpha ? RGBA.(out, α) : out
end



"""
    batch_auto_level_intensity(; tail=0.005)

Apply `auto_level_intensity` to all `.png` images in the current directory.

For each file:
- Load image.
- Apply intensity leveling with the given quantile `tail`.
- Save result using a filename with `_AUTO` inserted before `.png`.

Example:
`image.png` → `image_AUTO.png`

Useful for processing large batches of DAS-derived images.

# Keyword arguments
- `tail`: Fraction of pixels clipped at low/high ends before stretching (default 0.005).

# Side effects
- Writes new files to disk in the same directory.
- Prints progress messages.
"""
function batch_auto_level_intensity(; tail=0.005)
    files = filter(f -> endswith(lowercase(f), ".png"), readdir())
    for f in files
        println("Processing $f ...")
        img  = load(f)
        img2 = auto_level_intensity(img; tail)
        outname = replace(f, r"\.png$"i => "_AUTO.png")
        save(outname, img2)
        println(" → Saved $outname")
    end
end



# ---- 4. Image (Data) Augmentation --------------------------------------------------------
#   These functions serve to aument the pre-processed data in a pseudo-random way

"""
    das_image_data_aumentation(dDAS, zs_norm, filename; it=10)

Perform random data augmentation on a normalized DAS image matrix.

This routine generates `it` augmented versions of the input image by applying
random transformations that mimic realistic variations in DAS recordings.
Each augmented image is then saved using a color scale via `save_fig`.

Augmentations applied (each chosen independently with 50% probability):

1. **Polarity reversal**
   - Multiply the image by -1 (simulates phase inversion).

2. **Channel flipping**
   - Reverse spatial axis (fiber read backwards).

3. **Time shifting**
   - Circularly shift rows by a random amount (0.05–0.95 of image height).

4. **Time stretching (cropping)**
   - Randomly remove 1–10% of rows from top and bottom.

5. **Spatial channel dropout (cropping)**
   - Randomly remove 0–15% of columns from left and right.

The function encodes the augmentation operations performed as a 5-digit
string (`r1r2r3r4r5`) in the saved filename.

# Arguments
- `dDAS`:      DAS data object (used by `save_fig` for annotation/metadata).
- `zs_norm`:   Normalized DAS image matrix (2D array).
- `filename`:  Base filename of the original DAS record.
- `it`:        Number of augmented images to generate (default 10).

# Output
- Saves augmented figures in three colormaps: `:RdBu_9`, `:viridis`, `:grays`.
- Filenames include a timestamp and operator code.

# Notes
- No return value; this function performs disk I/O side effects.
- Designed for increasing training data diversity for ML classification.
"""
function das_image_data_aumentation(dDAS, zs_norm, filname; it=10)
    for i = 1:it
        timest = now()                  # unique timestamp for filenames
        zs = copy(zs_norm)              # work on a fresh copy

        # --- Polarity inversion (50% chance) ---
        r1 = rand(0:1)
        if r1 == 1
            zs = -zs
        end

        # --- Flip cable direction (reverse channel order) ---
        r2 = rand(0:1)
        if r2 == 1
            zs = zs[:, end:-1:1]
        end

        # --- Time shift (circular shift in rows) ---
        r3 = rand(0:1)
        if r3 == 1
            samples_das = floor(Int, size(zs,1) * rand(0.05:0.01:0.95))  # shift amount
            zs = [zs[samples_das:end, :]; zs_norm[1:samples_das-1, :]]
        end

        # --- Time stretching: remove a % of rows at top/bottom ---
        r4 = rand(0:1)
        if r4 == 1
            randrand1 = rand(1:10)          # 1–10%
            randrand2 = rand(1:10)          # 1–10%
            nrows = size(zs, 1)
            cut1 = floor(Int, nrows * randrand1 / 100)
            cut2 = floor(Int, nrows * randrand2 / 100)
            zs = zs[cut1+1 : nrows-cut2, :]  # crop rows
        end

        # --- Spatial dropout: remove a % of columns left/right ---
        r5 = rand(0:1)
        if r5 == 1
            randrand1 = rand(0:15)          # 0–15%
            randrand2 = rand(0:15)
            ncols = size(zs, 2)
            cut1 = floor(Int, ncols * randrand1 / 100)
            cut2 = floor(Int, ncols * randrand2 / 100)
            zs = zs[:, cut1+1 : ncols-cut2]  # crop columns
        end

        # Operator code used in filenames
        operator = "$r1$r2$r3$r4$r5"

        # --- Save augmented images in several color maps ---
        save_fig(dDAS, zs, :RdBu_9, "RdBu_$(operator)_$(timest)", filname)
        save_fig(dDAS, zs, :viridis, "viridis_$(operator)_$(timest)", filname)
        save_fig(dDAS, zs, :grays,   "grays_$(operator)_$(timest)", filname)
    end
end


# ---- 5. Figure Making --------------------------------------------------------
#  These functions are just to handdle figures

"""
    save_fig(dDAS, zs_blur, cm, suffix, filename)

Save a 2D DAS data matrix (`zs_blur`) as a 224×224 pixel heatmap image
using a specified colormap and filename suffix.

This helper function is used to generate training images for ML models
(e.g., ResNet), ensuring consistent figure size, formatting, and removal
of axes/spines for a clean image.

# Arguments
- `dDAS`: DAS data structure containing at least a `.name` field used for output filenames.
- `zs_blur`: 2D matrix to plot (preprocessed DAS data, e.g., blurred/filtered).
- `cm`: A Makie colormap symbol (e.g., `:viridis`, `:RdBu_9`, `:grays`).
- `suffix`: String appended to the base filename before `.png`.
- `filename`: Original DAS filename (only used for printed progress messages).

# Function behavior
- Computes color limits using ±3σ of the input matrix.
- Plots the matrix using a heatmap with:
    - fixed size 224×224 (to mimic ResNet input dimensions)
    - no axes, ticks, labels, or spines
    - no interpolation
- Saves output as:
      dDAS.name * "_" * suffix * ".png"

# Output
- Writes the image to disk.
- Prints a styled message confirming completion.

# Notes
- This function intentionally strips all plot decorations so that the
  resulting PNG is purely an image for ML ingestion.
"""
function save_fig(dDAS, zs_blur, cm, suffix, filename)
    climit = 3 * std(zs_blur)   # define symmetric color scaling around 0

    # Create a compact Makie figure (224×224 px)
    fig = Figure(figure_padding=0, size=(224, 224), aspect=DataAspect())

    # Plot as a heatmap (note: using zs_blur directly, no time/offset axes)
    ihm = heatmap(
        fig[1, 1],
        zs_blur,
        colormap = cm,
        colorrange = (-climit, climit),
        interpolate = false
    )

    # Remove axes, ticks, spines for a clean ML-ready image
    hidespines!(ihm.axis)
    hidedecorations!(ihm.axis)

    # Ensure tight layout
    fig.content[1,1] = ihm.axis
    rowgap!(fig.layout, 0)
    colgap!(fig.layout, 0)

    # Save figure with given suffix
    save(dDAS.name * "_" * suffix * ".png", fig; px_per_unit=1)

    # Console feedback
    printstyled("\nDone with $filename ($suffix)\n",
                color=:blue, bold=true, blink=true)
end



"""
    figs2dirs(; srcdir::AbstractString = ".")

Organize PNG figure files into timestamped subdirectories based on their
colormap name.

This function scans the directory `srcdir` (default: current directory)
and moves all `.png` files into a new directory structure:

    FIGS_YYYYMMDD_HHMMSS/
        ├── grays/
        ├── RdBu/
        └── viridis/

Images are routed to subfolders based on matching substrings in the filename:
- `"grays"`   → `grays/`
- `"rdbu"` or `"rdBu"` (case-insensitive) → `RdBu/`
- `"viridis"` → `viridis/`

Useful after generating large numbers of DAS images with different colormaps.

# Keyword Arguments
- `srcdir`: Directory to scan for `.png` files (default: `"."`).

# Returns
- The path to the newly created master directory.

# Side Effects
- Creates directories.
- Moves files on disk.
"""
function figs2dirs(; srcdir::AbstractString=".")
    # 1. Create master directory with timestamp
    thistime  = Dates.format(now(), "yyyymmdd_HHMMSS")
    masterdir = joinpath(srcdir, "FIGS_$thistime")

    # 2. Define subdirectories
    subdirs = ["grays", "RdBu", "viridis"]
    for sd in subdirs
        mkpath(joinpath(masterdir, sd))   # create each folder
    end

    # 3. Move PNG files into corresponding subdirectories
    for f in readdir(srcdir)
        filepath = joinpath(srcdir, f)
        if isfile(filepath) && endswith(f, ".png")
            lowerf = lowercase(f)         # case-insensitive matching
            if occursin("grays", lowerf)
                mv(filepath, joinpath(masterdir, "grays", f))
            elseif occursin("rdBu", f) || occursin("rdbu", lowerf)
                mv(filepath, joinpath(masterdir, "RdBu", f))
            elseif occursin("viridis", lowerf)
                mv(filepath, joinpath(masterdir, "viridis", f))
            end
        end
    end

    println("✅ All figures organized into $masterdir")
    return masterdir
end



"""
    noise2dirs(; srcdir::AbstractString = ".")

Organize PNG files related to noise examples into a timestamped directory
structure similar to `figs2dirs`, but placing them under a folder named:

    NOISE_YYYYMMDD_HHMMSS/

This is useful for separating noise-only DAS images after preprocessing.

Subfolder logic is identical to `figs2dirs`, sorting by colormap name:
- `"grays"`   → `grays/`
- `"rdbu"`    → `RdBu/`
- `"viridis"` → `viridis/`

# Keyword Arguments
- `srcdir`: Directory to scan for `.png` noise images (default `"."`).

# Returns
- The path to the newly created noise directory.

# Side Effects
- Creates directories.
- Moves files on disk.
"""
function noise2dirs(; srcdir::AbstractString=".")
    # 1. Create master directory with timestamp
    thistime  = Dates.format(now(), "yyyymmdd_HHMMSS")
    masterdir = joinpath(srcdir, "NOISE_$thistime")

    # 2. Define subdirectories
    subdirs = ["grays", "RdBu", "viridis"]
    for sd in subdirs
        mkpath(joinpath(masterdir, sd))
    end

    # 3. Move PNG files into corresponding subdirectories
    for f in readdir(srcdir)
        filepath = joinpath(srcdir, f)
        if isfile(filepath) && endswith(f, ".png")
            lowerf = lowercase(f)
            if occursin("grays", lowerf)
                mv(filepath, joinpath(masterdir, "grays", f))
            elseif occursin("rdBu", f) || occursin("rdbu", lowerf)
                mv(filepath, joinpath(masterdir, "RdBu", f))
            elseif occursin("viridis", lowerf)
                mv(filepath, joinpath(masterdir, "viridis", f))
            end
        end
    end

    println("✅ All Noise organized into $masterdir")
    return masterdir
end

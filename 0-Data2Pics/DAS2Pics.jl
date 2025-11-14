#!/usr/bin/env julia

#using NCDatasets, DASVader, CairoMakie, NanoDates, Dates, StatsBase
# include("get_bands.jl")
# include("fkpeak.jl")
# include("auto_level_intensity.jl")
# include("das_image_data_aumentation.jl")
# include("figs2dirs.jl")

include("Image-Signal-Processing.jl")


ncfiles = filter(f -> endswith(f, ".nc"), readdir())

for i in eachindex(ncfiles)

    ds = NCDataset(ncfiles[i], "r")

    # --- Data ---
    data = ds["StrainRate"].var[:, :]'   # transpose: dim1 = distance, dim2 = time

    offset = range(
        start  = ds["offset"].var[:][1],
        step   = mean(diff(ds["offset"].var[:])),
        length = size(ds["offset"].var[:], 1)
    )

    # --- Time ---
    time_raw = ds["time"].var[:]                # raw numbers from NetCDF
    Δt   = mean(diff(time_raw)) / 1e6           # step in seconds
    t0   = first(time_raw) / 1e6
    tend = last(time_raw) / 1e6
    timee = t0:Δt:tend                          # StepRangeLen (good for iDAS)

    # Parse reference date from units
    units = ds["time"].attrib["units"]
    m = match(r"(microseconds|milliseconds|seconds) since (.*)", units)
    if isnothing(m)
        error("Could not parse time units: $units")
    end
    scale = m.captures[1]
    ref   = NanoDate(m.captures[2])

    factor = scale == "microseconds"  ? 1 :
             scale == "milliseconds" ? 1_000 :
             scale == "seconds"      ? 1_000_000 :
             error("Unhandled time unit: $scale")

    times = ref .+ Microsecond.(time_raw .* (1_000_000 ÷ factor))

    nd1, nd2  = times[1], times[end]
    nanostep  = times[2] - times[1]
    nanorange = nd1:nanostep:nd2
    htime     = DateTime.(nanorange)

    printstyled("\nGot time and space axis\n", color=:green)

    # --- Attributes (empty defaults) ---
    atrib = attb(Int8[], Int32[], Int32[], Int32[], Float64[], Int32[], Int32[],
                 Float64[], "", Float64[], Int8[], "", 0.0, Int8[],
                 Int32[], Int32[], Float64[], Int32[])

    #GET THE FILE NAME AND SAVE THE DATA TO iDAS format
    filenameC = ncfiles[i]
    dDAS = iDAS(data, timee, htime, offset, atrib, filenameC)


### PROCESS THE DAS data
# Preprocess
    ppdas!(dDAS; rmean=true, rtrend=true, taper=false, w=0.01, f=:hanning)
    # norm to 1 each channel
       normdas!(dDAS; style="channel")
# Get the bands to filter the data
   dbmin, dbmax, freqs, SNR_med = get_dp(dDAS, seglen = 4096)
   dbmin=max(1,dbmin)
#Filter the data
    bpdas!(dDAS; f1=dbmin, f2=dbmax, poles=4)
# Pre process again
    ppdas!(dDAS; rmean=true, rtrend=true, taper=false, w=0.01, f=:hanning)
normdas!(dDAS; style="channel")
#
# Get the peak frequency and wavenumber of the FK spectrum.
Δx=dDAS.offset[2]-dDAS.offset[1]
Δt=dDAS.time[2]-dDAS.time[1]
f_pk, k_pk, f_ax, k_ax, S = fkpeak(dDAS.data, Δx, Δt, dbmin=dbmin, dbmax=dbmax)
#### APPLY GAUSS FILTER
zs_blur = fkgaussfilter(dDAS.data, Δx, Δt, f_pk, k_pk; relwidth=0.1)

#Normalize the blured matrix
col_max = maximum(abs, zs_blur; dims=1)
col_max[col_max .== 0] .= 1   # avoid division by zero
zs_norm = zs_blur ./ col_max

#### Make the figures
    save_fig(dDAS, zs_norm,:RdBu_9, "RdBu_0", filenameC)
    save_fig(dDAS, zs_norm,:viridis, "viridis_0", filenameC)
    save_fig(dDAS, zs_norm,:grays, "grays_0", filenameC)

 #####################################################
 # DATA_AUGMENTATION We try to do data augmentation
 #####################################################
das_image_data_aumentation(dDAS, zs_norm, filenameC; it=9)
end
#################################################
# Improve THE INTENSITIES in the figures
batch_auto_level_intensity()
# To dirs
figs2dirs()

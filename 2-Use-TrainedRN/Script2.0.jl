# ================================================================
#  DASight – Script 2 (V 2.0)
#  Author: Mariano Arnaiz
#  Description:
#  Run trained ResNet from Script 1 on new images to Classify them
#  Note: Images must the properly pre-processed (see Script 0)
# ================================================================

# ---- 0.1 Modules ----------------------------------------------------
using Flux, Metalhead
using Images, ImageIO, FileIO
using MLUtils   # for flatten
using BSON: @load

# ---- constants used during training ----
const μ = Float32[0.485, 0.456, 0.406]
const σ = Float32[0.229, 0.224, 0.225]
img_size = (224,224)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
@load "best_resnet18.bson" model classes

# ---------------------------------------------------------
# Preprocess ONE IMAGE — exactly like training
# ---------------------------------------------------------
"""
    preprocess_image(path::String) -> Array{Float32,3}

Load and preprocess a single DAS image so it is compatible with the
ResNet-based classifier.

This function applies the *exact same preprocessing steps used during training*:

1. Loads an image from disk.
2. Resizes it to the expected input size (`img_size`).
3. Converts the image to `Float32` in **channel-first (C×H×W)** format.
4. Ensures it has exactly 3 channels:
   - Grayscale images are repeated into RGB.
   - Images with an alpha channel are truncated to RGB.
5. Normalizes pixel values using ImageNet statistics (`μ` and `σ`).

# Arguments
- `path::String`: Path to the image file.

# Returns
- `Array{Float32,3}`: Preprocessed image tensor with shape `(3, H, W)`.

# Notes
- This ensures consistency between training and inference.
- `img_size`, `μ`, and `σ` must be defined globally or imported.
"""
function preprocess_image(path)
    # 1. Load image from file
    img = load(path)

    # 2. Resize to model input size (e.g., 224×224)
    img = imresize(img, img_size)

    # 3. Convert to Float32 and switch to channel-first format (C×H×W)
    img = Float32.(channelview(img))

    # 4. Handle channel mismatches:
    #    - grayscale (1 channel) → repeat into 3 channels
    size(img,1) == 1 && (img = repeat(img, 3, 1, 1))

    #    - RGBA (4 channels) → drop the alpha channel
    size(img,1) == 4 && (img = img[1:3, :, :])

    # 5. Normalize using ImageNet mean and std (same as during training)
    img = (img .- μ) ./ σ

    return img
end


# ---------------------------------------------------------
# Classify all images in a directory
# ---------------------------------------------------------
"""
    classify_dir(dir::String, model, classes) -> Array{Any,2}

Classify all valid image files inside a directory using a trained ResNet model.

This function:
1. Reads all files in `dir`.
2. Filters only valid image extensions (`.png`, `.jpg`, `.jpeg`).
3. Applies the **same preprocessing** used during training:
   - Resize to 224×224
   - Convert to Float32 and channel-first (C×H×W)
   - Ensure 3-channel RGB input
   - Normalize using ImageNet mean (`μ`) and std (`σ`)
4. Runs inference using the provided `model`.
5. Records:
   - filename
   - predicted class (from `classes`)
   - confidence score (softmax probability)

# Arguments
- `dir::String`: Path to directory containing images to classify.
- `model`: Trained ResNet-based classifier.
- `classes::Vector{String}`: Vector mapping class indices → class names.

# Returns
- `Array{Any,2}`: An `N × 3` matrix where N = number of files in `dir`.
  Columns:
    1. `String`: filename
    2. `String`: predicted class
    3. `Float64`: confidence score

# Notes
- The array contains `Any` because columns have mixed types.
- Files are processed in directory order.
"""
function classify_dir(dir::String, model, classes)
    # Preallocate an N×3 heterogeneous matrix
    predictions = Array{Any}(undef, size(readdir(dir), 1), 3)

    # Valid image extensions
    valid_ext = [".png", ".jpg", ".jpeg"]

    i = 0  # row counter for predictions

    for file in readdir(dir)
        ext = lowercase(splitext(file)[2])
        i += 1

        # Only process valid images
        if ext ∈ valid_ext
            # --- 1. Load and preprocess image (same as training) ---
            img = load(joinpath(dir, file))
            img = imresize(img, (224, 224))
            img = Float32.(channelview(img))  # CHW format

            # Grayscale → RGB
            size(img,1) == 1 && (img = repeat(img, 3, 1, 1))

            # Drop alpha channel
            size(img,1) == 4 && (img = img[1:3, :, :])

            # Normalize (ImageNet statistics)
            img = (img .- μ) ./ σ

            # Model expects WHCN format → add batch dimension
            img = cat(img, dims=4)
            img = permutedims(img, (2, 3, 1, 4))

            # --- 2. Inference ---
            y_hat = model(img)

            # Top-1 prediction
            idx = argmax(y_hat)
            prob = softmax(y_hat)[idx]

            # --- 3. Save results ---
            predictions[i, 1] = file
            predictions[i, 2] = classes[idx]
            predictions[i, 3] = prob
        end
    end

    return predictions
end

# ---------------------------------------------------------
# Run inference
# ---------------------------------------------------------
dir = "secret_test_set"
preds = classify_dir(dir, model, classes)

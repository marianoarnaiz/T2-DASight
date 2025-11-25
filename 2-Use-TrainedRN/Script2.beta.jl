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
function preprocess_image(path)
    img = load(path)
    img = imresize(img, img_size)             # resize
    img = Float32.(channelview(img))          # CHW

    # convert grayscale → RGB
    size(img,1) == 1 && (img = repeat(img, 3,1,1))

    # drop alpha
    size(img,1) == 4 && (img = img[1:3,:,:])

    # normalize (same as training)
    img = (img .- μ) ./ σ

    return img
end

# ---------------------------------------------------------
# Classify all images in a directory
# ---------------------------------------------------------
function classify_dir(dir::String, model, classes)
    predictions = Array{Any}(undef, size(readdir(dir), 1), 3)
    valid_ext = [".png", ".jpg", ".jpeg"]
    i=0
    for file in readdir(dir)
        ext = lowercase(splitext(file)[2])
        i=i+1
        if ext ∈ valid_ext
            img = load(joinpath(dir, file))
            img = imresize(img, (224, 224))                 # resize
            img = Float32.(channelview(img))                # CHW
            # convert grayscale -> RGB or drop alpha if necessary
            size(img,1) == 1 && (img = repeat(img, 3,1,1))
            size(img,1) == 4 && (img = img[1:3,:,:])
            # normalize
            img = (img .- μ) ./ σ

            img = cat(img, dims=4);  # Concatenate training images along 4th dim
            img = permutedims(img, (2, 3, 1, 4));

            y_hat = model(img)
            idx = argmax(y_hat)
            prob=softmax(y_hat)[idx]

            predictions[i,1] = file
            predictions[i,2] = classes[idx]
            predictions[i,3] = prob
        end
    end
    return predictions
end

# ---------------------------------------------------------
# Run inference
# ---------------------------------------------------------
dir = "secret_test_set"
preds = classify_dir(dir, model, classes)

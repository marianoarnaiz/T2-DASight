# ================================================================
#  DASight – Initial Script
#  Author: Mariano Arnaiz
#  Description:
#    Loads DAS image data, splits it into training and validation sets,
#    and trains a ResNet-18 model for event classification.
# ================================================================

# ---- 0.1 Modules ----------------------------------------------------
using Flux, Metalhead, Statistics, Random # Core machine learning and neural network tools
using Images, ImageIO, FileIO # Image loading and preprocessing
using Flux: DataLoader, onehotbatch, params # Utilities for data handling and saving model parameters
using BSON: @save

# ---- 0.2 Define Constants ------------------------------------------
# ImageNet normalization statistics (mean and std for each RGB channel)
# Used to normalize input images to match the ResNet-18 pretraining setup.
const μ = Float32[0.485f0, 0.456f0, 0.406f0] # mean
const σ = Float32[0.229f0, 0.224f0, 0.225f0] # std

# ---- 0.3 Define Functions ----------------------------------------------------

"""
    load_images_and_labels(data_dir::String, classes::Vector{String}) -> (X, y)

Loads and preprocesses labeled image data from a directory structure.

Each subfolder in `data_dir` should correspond to one class in `classes`
and contain `.png`, `.jpg`, or `.jpeg` images.

For each image:
- Resizes to `(224, 224)` to match ResNet input.
- Converts to Float32 and channel-first format (C×H×W).
- Expands grayscale to RGB or removes alpha channels if present.
- Normalizes pixel values using ImageNet mean (`μ`) and std (`σ`).

# Returns
- `X` : Vector of normalized image tensors (`Array{Float32,3}`).
- `y` : Vector of integer class indices corresponding to `classes`.
"""
function load_images_and_labels(data_dir, classes)
    X, y = Vector{Array{Float32,3}}(), Int[]
    valid_ext = [".png", ".jpg", ".jpeg"]
    for (label_idx, class) in enumerate(classes)
        for file in readdir(joinpath(data_dir, class))
            ext = lowercase(splitext(file)[2])
            if ext ∈ valid_ext
                img = load(joinpath(data_dir, class, file))
                img = imresize(img, (224, 224))                 # resize
                img = Float32.(channelview(img))                # CHW
                # convert grayscale -> RGB or drop alpha if necessary
                size(img,1) == 1 && (img = repeat(img, 3,1,1))
                size(img,1) == 4 && (img = img[1:3,:,:])
                # normalize
                img = (img .- μ) ./ σ
                push!(X, img)
                push!(y, label_idx)
            end
        end
    end
    return X, y
end


"""
    accuracy(model, loader) -> Float64

Computes classification accuracy for a trained model on a dataset.

Iterates over a `DataLoader`, performs forward inference on each batch,
and compares predicted and true class indices.

# Arguments
- `model`  : Flux model (e.g. ResNet-18).
- `loader` : `DataLoader` object containing `(x, y)` batches.

# Returns
- Mean classification accuracy as a `Float64` between 0 and 1.
"""
function accuracy(m, loader)
    total_correct = 0
    total_seen = 0
    for (x, y) in loader
        preds = m(x)
        # convert CartesianIndex -> Int
        pred_labels = vec(map(ind -> ind[1], argmax(preds, dims=1)))
        total_correct += sum(pred_labels .== y)
        total_seen += length(y)
    end
    return total_correct / total_seen
end

# ---- 1. Data Loading --------------------------------------------------------
# Define the root directory containing the training data.
# Each subdirectory inside `data_dir` corresponds to one class.
data_dir = "train"

# List available class folders, ignoring hidden files (e.g. .DS_Store)
classes = filter(x -> !startswith(x, "."), readdir(data_dir))
println("Classes found: ", classes)

# Load all images and labels from the dataset
X, y = load_images_and_labels(data_dir, classes)
println("Loaded $(length(X)) images across $(length(classes)) classes.")

# ---- 2. train/test split -----------------------------------------------
# Shuffle the indices of all images to ensure randomness
# Compute the split index at 80% of the dataset
# Divide the indices into training and testing sets
n = length(X);
idx = shuffle(1:n);
split_at = Int(round(0.8 * n));
train_idx, test_idx = idx[1:split_at], idx[split_at+1:end];

# ---- 3. stack arrays into (C,H,W,N) batches ----------------------------
# Convert the image arrays into a single 4D tensor for training/testing:
#   - X[i] are individual images (H x W x C)
#   - We concatenate them along a new 4th dimension (batch dimension)
#   - Initially cat(..., dims=4) produces (C,H,W,N) but may need permuting
#   - permutedims reorders dimensions to match expected input format for the model
#     (here we go from CHW + batch -> WHCB, adjust as needed for your framework)
# Convert labels to Int32 for compatibility with loss functions

x_train = cat([X[i] for i in train_idx]..., dims=4);  # Concatenate training images along 4th dim
x_train = permutedims(x_train, (2, 3, 1, 4));         # Reorder dimensions if needed
y_train = Int32.(y[train_idx]);                       # Convert training labels to Int32

x_test  = cat([X[i] for i in test_idx]...,  dims=4);  # Same for testing images
x_test  = permutedims(x_test,  (2, 3, 1, 4));         # Reorder dimensions
y_test  = Int32.(y[test_idx]);                        # Convert testing labels

# Print batch shapes for verification
println("train batch size: ", size(x_train), ", y_train length: ", length(y_train))
println("test batch size:  ", size(x_test),  ", y_test length: ", length(y_test))

# ---- 4. model: pretrained ResNet18 + small 2-class head -----------------
# Load a pretrained ResNet18 model (from Metalhead or Flux/Metalhead wrapper)
#   - The model is pretrained on ImageNet (1000 classes)
#   - `inchannels=3` ensures input has 3 color channels (RGB)
resnet = ResNet(18; pretrain=true, inchannels=3);   # or Metalhead.resnet18(pretrained=true)

# Append a new Dense layer to convert the 1000-dimensional ResNet output
# into 2 classes for our specific classification task
#   - Chain allows stacking multiple layers into a single model
#   - The final output of the model will be logits of size 2 per sample
model = Chain(resnet, Dense(1000, 2));

# ---- 5. loss, optimizer, accuracy -------------------------------------
# Define the loss function for 2-class classification:
#   - Flux.logitcrossentropy expects raw logits (no softmax)
#   - onehotbatch(y, 1:2) converts integer labels to one-hot vectors
loss(m, x, y) = Flux.logitcrossentropy(m(x), onehotbatch(y, 1:2));

# Choose the optimizer: Adam with learning rate 1e-3
#   - A smaller learning rate (e.g., 1e-5) could be used for fine-tuning
opt = Adam(1e-3);

# Get the parameters of the model for optimization
ps = params(model);

# ---- 6. DataLoaders ----------------------------------------------------
# Wrap training and testing data into DataLoaders for mini-batch iteration
#   - batchsize=32 specifies the number of samples per batch
#   - shuffle=true ensures training batches are randomized each epoch
#   - shuffle=false for test_loader to preserve order (no randomness needed)
train_loader = DataLoader((x_train, y_train), batchsize=32, shuffle=true);
test_loader  = DataLoader((x_test,  y_test),  batchsize=32, shuffle=false);

# ---- 7. training loop --------------------------------------------------
best_acc = 0.0                     # Keep track of best test accuracy
Epochs = 5                         # ← Increase epochs here to train longer
bad_epochs = 0                      # Counter for early-stopping-like logic (not used here)

# Arrays to store metrics for each epoch
train_loss_epoch = zeros(Epochs)   # Training loss per epoch
test_loss_epoch  = zeros(Epochs)   # Test/validation loss per epoch
acc_train_epoch  = zeros(Epochs)   # Training accuracy per epoch
acc_test_epoch   = zeros(Epochs)   # Test accuracy per epoch

# Loop over all epochs
for epoch in 1:Epochs
    batch_num = 0                   # Counter for batches
    running_loss = 0.0              # Accumulate loss over batches

    # Loop over mini-batches from the training DataLoader
    for (x, y) in train_loader
        batch_num += 1

        # Compute gradients of loss w.r.t model parameters
        gs = gradient(ps) do
            l = loss(model, x, y)  # Compute batch loss
            running_loss += l       # Accumulate running loss for printing
            return l
        end

        # Update model parameters using optimizer
        Flux.Optimise.update!(opt, ps, gs)

        # Print average batch loss every 10 batches
        if batch_num % 10 == 0
            println("  batch $batch_num  loss=$(round(running_loss / batch_num, digits=5))")
        end
    end

    # Compute full-epoch metrics after all batches
    train_loss = loss(model, x_train, y_train)   # Full training set loss
    test_loss  = loss(model, x_test, y_test)     # Full test set loss
    acc_train  = accuracy(model, train_loader)   # Training accuracy over batches
    acc_test   = accuracy(model, test_loader)    # Test accuracy over batches

    # Store metrics for plotting or analysis
    train_loss_epoch[epoch] = train_loss
    test_loss_epoch[epoch]  = test_loss
    acc_train_epoch[epoch]  = acc_train
    acc_test_epoch[epoch]   = acc_test

    # Print epoch summary
    println("epoch $epoch  train_loss=$(round(train_loss,digits=5))  test_loss=$(round(test_loss,digits=5))  acc_train=$(round(acc_train,digits=3))  acc_test=$(round(acc_test,digits=3))")
end

# ---- 8. Print the evolution into a matrix ----------------------------------------------------
# Combine all tracked metrics into a single matrix for easy viewing or export
# Columns are:
#   1. train_loss_epoch
#   2. test_loss_epoch
#   3. acc_train_epoch
#   4. acc_test_epoch
# Each row corresponds to one epoch
Train_Results = [train_loss_epoch  test_loss_epoch  acc_train_epoch  acc_test_epoch]

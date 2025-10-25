# Calibrated Quantization Implementation

## Overview

This implementation modifies the `computeQuantized()` function in `Convolutional_new.cpp` to use pre-calculated calibration statistics from `calibration_stats.json` instead of computing quantization parameters at runtime.

## Key Changes

### 1. **Calibration Statistics Loading**
- Automatically loads `calibration_stats.json` from various possible paths
- Parses Si (input scale) and zi (zero point) values for each layer
- Falls back to runtime calculation if calibration file not found

### 2. **Layer Identification**
- Uses a static counter (`conv_layer_count`) to identify layers in order:
  - First call: `conv2d` 
  - Second call: `conv2d_1`
  - Third call: `conv2d_2`
  - And so on...

### 3. **Performance Improvements**
- Eliminates expensive min/max calculations on input data
- Uses pre-calculated Si and zi values from calibration
- Still calculates weight scale (Sw) at runtime for accuracy

### 4. **Better Logging**
- Uses Utils.h logging functions (`logInfo`, `logDebug`, `logError`)
- More structured debug output
- Reduced console spam

## Usage

### Individual Layer Testing (Default Mode)
```cpp
#include "Convolutional.h"

// For testing individual layers with raw image input
// (This is what the current test framework does)
ML::enableLayerSpecificCalibration(false);  // Default mode

// Each layer test gets raw image input, so use "_input" calibration stats
layer0.computeQuantized(rawImageData);  // Uses "_input" stats
layer1.computeQuantized(rawImageData);  // Uses "_input" stats  
layer2.computeQuantized(rawImageData);  // Uses "_input" stats
```

### Full Inference Chain Mode
```cpp
#include "Convolutional.h"

// For running complete inference chains
ML::enableLayerSpecificCalibration(true);
ML::resetConvLayerCounter();

// Each layer gets output from previous layer
layer0.computeQuantized(rawImageData);      // Uses "_input" stats
layer1.computeQuantized(layer0.getOutput()); // Uses "conv2d" stats  
layer2.computeQuantized(layer1.getOutput()); // Uses "conv2d_1" stats
// etc.
```

### Calibration File Structure
The implementation expects `calibration_stats.json` in this format:
```json
{
  "conv2d": {
    "min": -4.014701843261719,
    "max": 1.6905503273010254, 
    "mean": -0.26804782335069444,
    "Si": 33.89691157098514,
    "zi": 9
  },
  "conv2d_1": {
    "Si": 5.105264582460136,
    "zi": -15
  },
  "_input": {
    "Si": 230.48977346980885,
    "zi": -103
  }
}
```

### File Search Order
The implementation searches for calibration stats in these locations:
1. `../../../SW/Lab3/Phase_I_Calibration/calibration_stats.json`
2. `../../SW/Lab3/Phase_I_Calibration/calibration_stats.json`  
3. `../SW/Lab3/Phase_I_Calibration/calibration_stats.json`
4. `SW/Lab3/Phase_I_Calibration/calibration_stats.json`
5. `calibration_stats.json`

## Benefits

### Accuracy Benefits
- **Consistent quantization**: Same parameters used during calibration and inference
- **Representative statistics**: Based on 1000+ calibration images, not single input
- **Layer-specific tuning**: Different Si/zi values optimized for each layer's activation range

### Performance Benefits  
- **Faster inference**: No runtime min/max calculations on input data
- **Reduced memory bandwidth**: Still uses int8 operations for core computation
- **Production ready**: Matches industry-standard quantization workflows

### Debugging Features
- **Layer tracking**: Know exactly which layer is being processed
- **Fallback support**: Gracefully handles missing calibration files
- **Detailed logging**: Track quantization parameters being used

## Expected Results

With proper calibration data, you should see:
- **High cosine similarity** (>95%) compared to fp32 baseline
- **Consistent accuracy** across different input images
- **Faster execution** due to eliminated runtime statistics calculation
- **Layer-appropriate quantization** with different scales per layer

## Troubleshooting

### High Zero Output Percentage (>90%)
This indicates severe quantization parameter mismatch:
```cpp
// WRONG: Using layer-specific stats for raw image input
ML::enableLayerSpecificCalibration(true);  
layer2.computeQuantized(rawImageData);  // Uses "conv2d_1" stats - WRONG!

// CORRECT: Use input stats for raw image data  
ML::enableLayerSpecificCalibration(false);
layer2.computeQuantized(rawImageData);  // Uses "_input" stats - CORRECT!
```

### Poor Cosine Similarity (<80%)
- **For individual layer tests**: Use `enableLayerSpecificCalibration(false)`
- **For full inference chains**: Use `enableLayerSpecificCalibration(true)` and reset counter
- Ensure input data type matches calibration stats being used

### "No calibration stats found for layer"
- Check that `calibration_stats.json` contains entries for all expected layers
- Verify layer naming matches the expected pattern (`conv2d`, `conv2d_1`, etc.)
- Make sure `_input` entry exists for raw image data

### "Could not find calibration_stats.json file"
- Ensure the file exists in one of the searched paths
- Check file permissions and path separators  
- The code will fall back to runtime calculation if file not found

### Cascading Accuracy Loss Through Network
This suggests wrong calibration mode:
- Layer 0: 99% ✅ → Layer 1: 70% ❌ → Layer 2: 0% ❌
- Solution: Check `enableLayerSpecificCalibration()` setting matches your use case

## Technical Details

### Mathematical Foundation
The calibrated approach uses:
- **Input quantization**: `quantized = round(Si * input) + zi`
- **Weight quantization**: `quantized = round(Sw * weight)` (still calculated at runtime)
- **Dequantization**: `output = (accumulator - zi*weight_sum) / (Si * Sw)`

### Memory Usage
- Temporary vectors for quantized inputs, weights, and biases
- Same memory footprint as original implementation
- Pre-calculated values reduce computation overhead

### Thread Safety
- Static variables used for layer counting and calibration data
- Not thread-safe if multiple inference sessions run concurrently
- Reset counter before each inference session for consistency
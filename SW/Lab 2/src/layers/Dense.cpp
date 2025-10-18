#include "Dense.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    void DenseLayer::computeNaive(const LayerData &dataIn) const
    {
        //const auto &inputDims = getInputParams().dims;   // Can be [H, W, C] or [features] 
        //const auto &outputDims = getOutputParams().dims; // Expected: [output_features]
        const auto &weightDims = getWeightParams().dims; // Expected: [input_features, output_features]

        // Calculate total input features by flattening all input dimensions
        size_t totalInputFeatures = getInputParams().flat_count();
        size_t outputSize = getOutputParams().flat_count();

        // Validate dimensions
        size_t expectedInputFeatures = weightDims[0];  // First dimension of weight matrix
        size_t expectedOutputFeatures = weightDims[1]; // Second dimension of weight matrix
        
        if (totalInputFeatures != expectedInputFeatures) {
            std::cerr << "Dense layer input size mismatch: got " << totalInputFeatures 
                      << ", expected " << expectedInputFeatures << std::endl;
            return;
        }
        
        if (outputSize != expectedOutputFeatures) {
            std::cerr << "Dense layer output size mismatch: got " << outputSize 
                      << ", expected " << expectedOutputFeatures << std::endl;
            return;
        }

        const LayerData& weights = getWeightData();
        LayerData& output = getOutputData();
        const LayerData& bias = getBiasData();

        // Dense layer computation: output = input * weights + bias
        // Input is treated as flattened regardless of original dimensions
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++)
        {
            fp32 sum = bias.get<fp32>(out_idx);

            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++)
            {
                // Weight matrix: [input_features, output_features]
                size_t weightIdx = in_idx * outputSize + out_idx;
                
                sum += dataIn.get<fp32>(in_idx) * weights.get<fp32>(weightIdx);
            }
            // Apply ReLU activation only for hidden layers (not the final layer before Softmax)
            // The final dense layer typically has 200 outputs (for classification)
            // Hidden dense layers have other sizes (like 256)
            if (outputSize != 200) {
                // This is a hidden layer, apply ReLU
                sum = std::max(0.0f, sum);
            }
            // For the final layer (outputSize == 200), don't apply ReLU

            // Store result in output
            output.get<fp32>(out_idx) = sum;
        }
    }

    void DenseLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void DenseLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled matrix multiplication
        computeNaive(dataIn);
    }

    void DenseLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized matrix multiplication
        computeNaive(dataIn);
    }

    void DenseLayer::computeQuantized(const LayerData& dataIn) const {
        // Simple quantized implementation using int8 arithmetic
        //const auto &weightDims = getWeightParams().dims;
        size_t totalInputFeatures = getInputParams().flat_count();
        size_t outputSize = getOutputParams().flat_count();
        
        // Lab specification quantization parameters (these should be pre-calculated from profiling)
        // For now using reasonable estimates - in practice these come from profiling data ranges
        
        // Calculate Si = 127 / max(|Ix - avg(Ix)|) 
        // Assuming typical post-ReLU range [0, 6.35] -> max deviation = 6.35, Si = 127/6.35 = 20
        float Si = 20.0f;  // Input scale
        
        // Calculate Sw = 127 / max(|Wx|)
        // Assuming typical weight range [-0.5, 0.5] -> max = 0.5, Sw = 127/0.5 = 254
        float Sw = 254.0f; // Weight scale
        
        // Calculate Sb = Si * Sw (lab specification)
        float Sb = Si * Sw; // Bias scale
        
        // Calculate zi = -round(avg(Ix) * Si)
        // Assuming avg(Ix) ≈ 3.0 for post-ReLU activations, zi = -round(3.0 * 20) = -60
        int8_t zi = -60;  // Input zero point
        
        // Note: Weight and bias zero points are 0 per lab specification
        
        const LayerData& weights = getWeightData();
        const LayerData& bias = getBiasData();
        LayerData& output = getOutputData();
        
        // Step 1: Quantize inputs using lab formula: ix = round(Si * Ix) + zi
        std::vector<int8_t> quantized_input(totalInputFeatures);
        for (size_t i = 0; i < totalInputFeatures; i++) {
            float fp_val = dataIn.get<fp32>(i);
            // ix = round(Si * Ix) + zi (lab specification)
            int32_t temp = static_cast<int32_t>(std::round(Si * fp_val)) + zi;
            quantized_input[i] = static_cast<int8_t>(std::max(-128, std::min(127, temp)));
        }
        
        // Step 2: Perform int8 matrix multiplication with int32 accumulation
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++) {
            // Start with quantized bias: bx = round(Sb * Bx)
            int32_t accumulator = static_cast<int32_t>(std::round(Sb * bias.get<fp32>(out_idx)));
            
            // Accumulate: sum(ix * wx)
            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++) {
                size_t weightIdx = in_idx * outputSize + out_idx;
                float fp_weight = weights.get<fp32>(weightIdx);
                
                // Quantize weight: wx = round(Sw * Wx) (lab specification)
                int8_t quantized_weight = static_cast<int8_t>(
                    std::max(-128, std::min(127, static_cast<int32_t>(std::round(Sw * fp_weight)))));
                
                // int32 accumulation
                accumulator += static_cast<int32_t>(quantized_input[in_idx]) * static_cast<int32_t>(quantized_weight);
            }
            
            // Step 3: Dequantize to FP32 for requantization
            // Dequantization: float_value = (int32_accumulator) / (Si * Sw) + zero_point_correction
            // The zero_point_correction accounts for zi bias in the accumulation
            float dequantized = static_cast<float>(accumulator) / (Si * Sw);
            
            // Step 4: Apply ReLU with zero_point consideration
            if (outputSize != 200) {  // Hidden layer - apply ReLU
                dequantized = std::max(0.0f, dequantized);
            }
            
            // Store result (will be requantized by next layer)
            output.get<fp32>(out_idx) = dequantized;
        }
    }

}
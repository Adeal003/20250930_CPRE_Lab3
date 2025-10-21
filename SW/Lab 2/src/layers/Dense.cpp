#include "Dense.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>
#include <cmath>

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
        // Debug: Verify quantized path is being taken
        std::cout << "[DEBUG] Dense computeQuantized() called" << std::endl;
        
        // Get dimensions
        size_t totalInputFeatures = getInputParams().flat_count();
        size_t outputSize = getOutputParams().flat_count();
        
        // Get data pointers
        const LayerData& weights = getWeightData();
        const LayerData& bias = getBiasData();
        LayerData& output = getOutputData();
        
        // ===== STEP 1: Calculate weight scale Sw =====
        // Sw = 127 / max(|Wx|)
        fp32 max_weight = 0.0f;
        for (size_t i = 0; i < totalInputFeatures * outputSize; i++) {
            fp32 abs_val = std::abs(weights.get<fp32>(i));
            if (abs_val > max_weight) {
                max_weight = abs_val;
            }
        }
        std::cout << "[DEBUG] Max weight value: " << max_weight << std::endl;
        fp32 Sw = 127.0f / max_weight;
        
        // ===== STEP 2: Calculate input scale Si and zero point zi =====
        // Si = 127 / max(|Ix - avg(Ix)|)
        // zi = -round(avg(Ix) * Si)
        
        // Calculate input average
        fp32 input_sum = 0.0f;
        for (size_t i = 0; i < totalInputFeatures; i++) {
            input_sum += dataIn.get<fp32>(i);
        }
        fp32 input_avg = input_sum / totalInputFeatures;
        
        // Find max deviation from average
        fp32 max_deviation = 0.0f;
        for (size_t i = 0; i < totalInputFeatures; i++) {
            fp32 deviation = std::abs(dataIn.get<fp32>(i) - input_avg);
            if (deviation > max_deviation) {
                max_deviation = deviation;
            }
        }
        
        // Calculate scale and zero point
        fp32 Si = 127.0f / max_deviation;
        i8 zi = static_cast<i8>(-std::round(input_avg * Si));
        
        // ===== STEP 3: Calculate bias scale =====
        // Sb = Si * Sw
        fp32 Sb = Si * Sw;
        
        // ===== STEP 4: Quantize inputs =====
        // ix = round(Si * Ix) + zi
        std::vector<i8> quantized_input(totalInputFeatures);
        for (size_t i = 0; i < totalInputFeatures; i++) {
            i32 temp = static_cast<i32>(std::round(Si * dataIn.get<fp32>(i))) + zi;
            // Clamp to int8 range
            quantized_input[i] = static_cast<i8>(std::max(-128, std::min(127, temp)));
        }
        
        // ===== STEP 5: Perform quantized computation =====
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++) {
            // Start with quantized bias: bx = round(Sb * Bx)
            i32 accumulator = static_cast<i32>(std::round(Sb * bias.get<fp32>(out_idx)));
            
            // Accumulate: sum(ix * wx)
            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++) {
                size_t weightIdx = in_idx * outputSize + out_idx;
                fp32 fp_weight = weights.get<fp32>(weightIdx);
                
                // Quantize weight: wx = round(Sw * Wx)
                i32 temp_weight = static_cast<i32>(std::round(Sw * fp_weight));
                i8 quantized_weight = static_cast<i8>(std::max(-128, std::min(127, temp_weight)));
                
                // int32 accumulation
                accumulator += static_cast<i32>(quantized_input[in_idx]) * static_cast<i32>(quantized_weight);
            }
            
            // ===== STEP 6: Apply ReLU in quantized space =====
            // ReLU: if accumulator < zi, set to zi (NOT zero!)
            if (outputSize != 200) {  // Hidden layer - apply ReLU
                if (accumulator < static_cast<i32>(zi)) {
                    accumulator = static_cast<i32>(zi);
                }
            }
            
            // ===== STEP 7: Dequantize back to fp32 =====
            // fp32_value = (int32_value - zi) / (Si * Sw)
            fp32 dequantized = static_cast<fp32>(accumulator - zi) / (Si * Sw);
            
            // Store result
            output.get<fp32>(out_idx) = dequantized;
        }
        
        // // Debug: Print first few output values
        // std::cout << "[DEBUG] First 3 quantized outputs: " 
        //           << output.get<fp32>(0) << ", " 
        //           << output.get<fp32>(1) << ", " 
        //           << output.get<fp32>(2) << std::endl;
    }

}  // namespace ML
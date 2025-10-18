#include "Softmax.h"

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

    void SoftmaxLayer::computeNaive(const LayerData &dataIn) const
    {
        //const auto &inputDims = getInputParams().dims;   // Expected: [batch, features] or just [features]
        //const auto &outputDims = getOutputParams().dims; // Expected: same as input

        // Get the number of elements to process
        size_t numElements = getInputParams().flat_count();
        
        LayerData& output = getOutputData();

        // Find the maximum value for numerical stability
        fp32 maxVal = -INFINITY;
        for (size_t i = 0; i < numElements; i++)
        {
            fp32 val = dataIn.get<fp32>(i);
            if (val > maxVal)
            {
                maxVal = val;
            }
        }

        // Compute exponentials and sum
        fp32 sumExp = 0.0f;
        for (size_t i = 0; i < numElements; i++)
        {
            fp32 expVal = std::exp(dataIn.get<fp32>(i) - maxVal);
            output.get<fp32>(i) = expVal;
            sumExp += expVal;
        }

        // Normalize by the sum
        for (size_t i = 0; i < numElements; i++)
        {
            output.get<fp32>(i) = output.get<fp32>(i) / sumExp;
        }
    }

    void SoftmaxLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void SoftmaxLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled processing
        computeNaive(dataIn);
    }

    void SoftmaxLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized softmax
        computeNaive(dataIn);
    }

    void SoftmaxLayer::computeQuantized(const LayerData& dataIn) const {
        // Softmax must operate on dequantized (FP32) values as per documentation
        size_t numElements = getInputParams().flat_count();
        LayerData& output = getOutputData();
        
        // Quantization parameters for input
        float input_scale = 1.0f / 127.0f;
        int8_t input_zero_point = 0;
        
        // Step 1: Quantize inputs to int8 first (simulating input from previous layer)
        std::vector<int8_t> quantized_input(numElements);
        for (size_t i = 0; i < numElements; i++) {
            float fp_val = dataIn.get<fp32>(i);
            int32_t temp = static_cast<int32_t>(std::round(input_scale * fp_val)) + input_zero_point;
            quantized_input[i] = static_cast<int8_t>(std::max(-128, std::min(127, temp)));
        }
        
        // Step 2: Dequantize back to FP32 for softmax computation
        std::vector<fp32> dequantized_input(numElements);
        for (size_t i = 0; i < numElements; i++) {
            // float_value = (int8_value - zero_point) * scale
            dequantized_input[i] = static_cast<float>(quantized_input[i] - input_zero_point) * input_scale;
        }
        
        // Step 3: Standard softmax computation on FP32 values
        // Find maximum for numerical stability
        fp32 maxVal = -INFINITY;
        for (size_t i = 0; i < numElements; i++) {
            if (dequantized_input[i] > maxVal) {
                maxVal = dequantized_input[i];
            }
        }

        // Compute exponentials and sum
        fp32 sumExp = 0.0f;
        for (size_t i = 0; i < numElements; i++) {
            fp32 expVal = std::exp(dequantized_input[i] - maxVal);
            output.get<fp32>(i) = expVal;
            sumExp += expVal;
        }

        // Normalize by the sum
        for (size_t i = 0; i < numElements; i++) {
            output.get<fp32>(i) = output.get<fp32>(i) / sumExp;
        }
        // Note: Softmax output is typically kept in FP32 for final classification
    }

}
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
        // Softmax per lab specs: "use the dequantized values (fp32) for the softmax function, 
        // your softmax function will remain unchanged"
        size_t numElements = getInputParams().flat_count();
        LayerData& output = getOutputData();
        
        // Use quantization parameters matching previous layer (Dense layer output)
        float Si = 20.0f;  // Should match previous layer's output quantization
        int8_t zi = -60;   // Should match previous layer's zero point
        
        // Step 1: Quantize inputs to int8 first (simulating quantized input from previous layer)
        std::vector<int8_t> quantized_input(numElements);
        for (size_t i = 0; i < numElements; i++) {
            float fp_val = dataIn.get<fp32>(i);
            // ix = round(Si * Ix) + zi (lab specification)
            int32_t temp = static_cast<int32_t>(std::round(Si * fp_val)) + zi;
            quantized_input[i] = static_cast<int8_t>(std::max(-128, std::min(127, temp)));
        }
        
        // Step 2: Dequantize back to FP32 for softmax computation
        // Dequantization: float_value = (int8_value - zero_point) * scale
        std::vector<fp32> dequantized_input(numElements);
        for (size_t i = 0; i < numElements; i++) {
            dequantized_input[i] = static_cast<float>(quantized_input[i] - zi) / Si;
        }
        
        // Step 3: Standard softmax computation on FP32 values (unchanged per lab specs)
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
        // Note: Softmax output remains in FP32 for final classification (lab specification)
    }

}
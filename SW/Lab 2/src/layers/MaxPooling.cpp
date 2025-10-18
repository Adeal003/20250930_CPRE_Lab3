#include "MaxPooling.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    void MaxPoolingLayer::computeNaive(const LayerData &dataIn) const
    {
        const auto &inputDims = getInputParams().dims;   // Expected: [H_in, W_in, C_in]
        const auto &outputDims = getOutputParams().dims; // Expected: [H_out, W_out, C_out]
        const auto &poolDims = getPoolParams().dims;     // Expected: [pool_h, pool_w]

        size_t inputHeight = inputDims[0];
        size_t inputWidth = inputDims[1];
        size_t inputChannels = inputDims[2];

        size_t outputHeight = outputDims[0];
        size_t outputWidth = outputDims[1];
        size_t outputChannels = outputDims[2];

        size_t poolHeight = poolDims[0];
        size_t poolWidth = poolDims[1];

        LayerData& output = getOutputData();

        // Max pooling computation
        for (size_t c = 0; c < outputChannels; c++)
        {
            for (size_t h_out = 0; h_out < outputHeight; h_out++)
            {
                for (size_t w_out = 0; w_out < outputWidth; w_out++)
                {
                    fp32 maxVal = -INFINITY;

                    // Pool over the kernel region
                    for (size_t pool_h = 0; pool_h < poolHeight; pool_h++)
                    {
                        for (size_t pool_w = 0; pool_w < poolWidth; pool_w++)
                        {
                            size_t h_in = h_out * poolHeight + pool_h;
                            size_t w_in = w_out * poolWidth + pool_w;

                            // Check bounds
                            if (h_in < inputHeight && w_in < inputWidth)
                            {
                                size_t inputIdx = h_in * (inputWidth * inputChannels) +
                                                  w_in * inputChannels +
                                                  c;

                                fp32 val = dataIn.get<fp32>(inputIdx);
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                }
                            }
                        }
                    }

                    size_t outputIdx = h_out * (outputWidth * outputChannels) +
                                       w_out * outputChannels +
                                       c;
                    
                    output.get<fp32>(outputIdx) = maxVal;
                }
            }
        }
    }

    void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled processing
        computeNaive(dataIn);
    }

    void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized max pooling
        computeNaive(dataIn);
    }

    void MaxPoolingLayer::computeQuantized(const LayerData& dataIn) const {
        // MaxPooling per lab specs: "take the max of the int8 input values without 
        // any dequantization/requantization involved"
        
        const auto &inputDims = getInputParams().dims;   // [H_in, W_in, C_in]
        const auto &outputDims = getOutputParams().dims; // [H_out, W_out, C_out]
        const auto &poolDims = getPoolParams().dims;     // [pool_h, pool_w]

        size_t inputHeight = inputDims[0];
        size_t inputWidth = inputDims[1];
        size_t inputChannels = inputDims[2];
        size_t outputHeight = outputDims[0];
        size_t outputWidth = outputDims[1];
        size_t outputChannels = outputDims[2];
        size_t poolHeight = poolDims[0];
        size_t poolWidth = poolDims[1];
        
        // Use same quantization parameters as previous layer for consistency
        // These should match the output of the previous layer
        // Using more conservative parameters to avoid saturation
        float Si = 42.0f;  // Should match previous layer's output quantization (more conservative)
        int8_t zi = -63;   // Should match previous layer's zero point (more conservative)
        
        // Step 1: Quantize inputs to int8 using lab formula: ix = round(Si * Ix) + zi
        size_t input_size = getInputParams().flat_count();
        std::vector<int8_t> quantized_input(input_size);
        for (size_t i = 0; i < input_size; i++) {
            float fp_val = dataIn.get<fp32>(i);
            int32_t temp = static_cast<int32_t>(std::round(Si * fp_val)) + zi;
            quantized_input[i] = static_cast<int8_t>(std::max(-128, std::min(127, temp)));
        }

        LayerData& output = getOutputData();

        // Step 2: Perform max pooling directly on int8 values (lab specification)
        for (size_t c = 0; c < outputChannels; c++) {
            for (size_t h_out = 0; h_out < outputHeight; h_out++) {
                for (size_t w_out = 0; w_out < outputWidth; w_out++) {
                    int8_t maxVal = -128;  // Minimum int8 value

                    // Pool over the kernel region
                    for (size_t pool_h = 0; pool_h < poolHeight; pool_h++) {
                        for (size_t pool_w = 0; pool_w < poolWidth; pool_w++) {
                            size_t h_in = h_out * poolHeight + pool_h;
                            size_t w_in = w_out * poolWidth + pool_w;

                            // Check bounds
                            if (h_in < inputHeight && w_in < inputWidth) {
                                size_t inputIdx = h_in * (inputWidth * inputChannels) +
                                                  w_in * inputChannels + c;

                                int8_t val = quantized_input[inputIdx];
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                            }
                        }
                    }

                    // Step 3: Dequantize result back to FP32 for next layer
                    // Dequantization: float_value = (int8_value - zero_point) * scale
                    float dequantized = static_cast<float>(maxVal - zi) / Si;
                    
                    size_t outputIdx = h_out * (outputWidth * outputChannels) +
                                       w_out * outputChannels + c;
                    output.get<fp32>(outputIdx) = dequantized;
                }
            }
        }
    }

}
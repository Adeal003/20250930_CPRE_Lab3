#include "Convolutional.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{
    // --- Begin Student Code ---
    // ASDFf
    // Compute the convolution for the layer data
    // Get dimensions from layer parameters
  
    // Perform convolution
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
        // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data

        const auto &inputDims = getInputParams().dims;   // [H, W, C_in]
        const auto &outputDims = getOutputParams().dims; // [H_out, W_out, C_out]
        const auto &weightDims = getWeightParams().dims; // [K_H, K_W, C_in, C_out]

        size_t U = 1; // Stride

       // size_t H = inputDims[0];
        size_t W = inputDims[1];
        size_t C = inputDims[2];

        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];

        size_t R = weightDims[0];
        size_t S = weightDims[1];

        for (size_t p = 0; p < P; p++)
        {
            for (size_t q = 0; q < Q; q++)
            {
                for (size_t m = 0; m < M; m++)
                {
                    fp32 result = 0.0f;
                    
                    // Perform the convolution sum
                    // o[p][q][m] = sum_{c,r,s} i[U*p+r][U*q+s][c] * f[r][s][c][m] + b[m]
                    for (size_t c = 0; c < C; c++)
                    { // Input channel
                        for (size_t r = 0; r < R; r++)
                        { // Kernel height
                            for (size_t s = 0; s < S; s++)
                            { // Kernel width
                                // Input coordinates
                                size_t input_h = U * p + r;
                                size_t input_w = U * q + s;
                                
                                // Input index: [input_h, input_w, c]
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                
                                // Weight index: [r, s, c, m]
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                
                                // Accumulate
                                result += dataIn.get<fp32>(input_idx) *
                                          getWeightData().get<fp32>(weight_idx);
                            }
                        }
                    }
                    // Add bias: b[m]
                    result += getBiasData().get<fp32>(m);
                    
                    // Apply ReLU activation
                    result = std::max(0.0f, result);
                    
                    // Output index: [p, q, m]
                    size_t output_idx = p * Q * M + q * M + m;
                    getOutputData().get<fp32>(output_idx) = result;
                }
            }
        }
    }

    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation with thread hints
        computeNaive(dataIn);
    }

    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation
        computeNaive(dataIn);
    }

    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation
        computeNaive(dataIn);
    }

    // Compute the convolution using quantized int8 arithmetic
    void ConvolutionalLayer::computeQuantized(const LayerData &dataIn) const
    {
        // Simple quantized convolution using int8 arithmetic
        const auto &inputDims = getInputParams().dims;   // [H, W, C_in]
        const auto &outputDims = getOutputParams().dims; // [H_out, W_out, C_out]
        const auto &weightDims = getWeightParams().dims; // [K_H, K_W, C_in, C_out]

        size_t U = 1; // Stride
        size_t W = inputDims[1];
        size_t C = inputDims[2];
        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];
        size_t R = weightDims[0];
        size_t S = weightDims[1];
        
        // Quantization parameters (assume they're set up)
        float input_scale = 1.0f / 127.0f;    // Si = 127 / max(|Ix - avg(Ix)|)
        float weight_scale = 1.0f / 127.0f;   // Sw = 127 / max(|Wx|)
        float bias_scale = input_scale * weight_scale;  // Sb = Si * Sw
        int8_t input_zero_point = 0;  // zi = -round(avg(Ix) * Si)
        
        // Step 1: Quantize inputs to int8
        size_t input_size = getInputParams().flat_count();
        std::vector<int8_t> quantized_input(input_size);
        for (size_t i = 0; i < input_size; i++) {
            float fp_val = dataIn.get<fp32>(i);
            // ix = round(Si * Ix) + zi
            int32_t temp = static_cast<int32_t>(std::round(input_scale * fp_val)) + input_zero_point;
            quantized_input[i] = static_cast<int8_t>(std::max(-128, std::min(127, temp)));
        }

        // Step 2: Perform quantized convolution
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    // Start with quantized bias (int32)
                    int32_t accumulator = static_cast<int32_t>(
                        std::round(bias_scale * getBiasData().get<fp32>(m)));
                    
                    // Perform the convolution sum in int8
                    for (size_t c = 0; c < C; c++) {
                        for (size_t r = 0; r < R; r++) {
                            for (size_t s = 0; s < S; s++) {
                                // Input coordinates
                                size_t input_h = U * p + r;
                                size_t input_w = U * q + s;
                                
                                // Input index: [input_h, input_w, c]
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                
                                // Weight index: [r, s, c, m]
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                
                                // Quantize weight: wx = round(Sw * Wx)
                                float fp_weight = getWeightData().get<fp32>(weight_idx);
                                int8_t quantized_weight = static_cast<int8_t>(
                                    std::max(-128, std::min(127, static_cast<int32_t>(std::round(weight_scale * fp_weight)))));
                                
                                // int32 accumulation: ix * wx
                                accumulator += static_cast<int32_t>(quantized_input[input_idx]) * 
                                              static_cast<int32_t>(quantized_weight);
                            }
                        }
                    }
                    
                    // Step 3: Dequantize back to FP32
                    // float_value = (int32_value - zero_point_offset) / (Si * Sw)
                    float dequantized = static_cast<float>(accumulator - input_zero_point * 0) / (input_scale * weight_scale);
                    
                    // Step 4: Apply ReLU activation
                    dequantized = std::max(0.0f, dequantized);
                    
                    // Store result
                    size_t output_idx = p * Q * M + q * M + m;
                    getOutputData().get<fp32>(output_idx) = dequantized;
                }
            }
        }
    }

} // namespace ML

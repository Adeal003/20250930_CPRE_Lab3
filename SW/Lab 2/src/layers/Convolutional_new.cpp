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
    // Compute the convolution using quantized int8 arithmetic
    void ConvolutionalLayer::computeQuantized(const LayerData &dataIn) const
    {
        // Debug: Verify quantized path is being taken
        std::cout << "[DEBUG] Conv computeQuantized() called" << std::endl;
        
        // Get dimensions
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
        
        // ===== STEP 1: Calculate weight scale Sw =====
        // Sw = 127 / max(|Wx|)
        size_t weight_size = getWeightParams().flat_count();
        fp32 max_weight = 0.0f;
        for (size_t i = 0; i < weight_size; i++) {
            fp32 abs_val = std::abs(getWeightData().get<fp32>(i));
            if (abs_val > max_weight) {
                max_weight = abs_val;
            }
        }
        fp32 Sw = 127.0f / max_weight;
        
        // ===== STEP 2: Calculate input scale Si and zero point zi =====
        // Si = 127 / max(|Ix - avg(Ix)|)
        // zi = -round(avg(Ix) * Si)
        
        size_t input_size = getInputParams().flat_count();
        
        // Calculate input average
        fp32 input_sum = 0.0f;
        for (size_t i = 0; i < input_size; i++) {
            input_sum += dataIn.get<fp32>(i);
        }
        fp32 input_avg = input_sum / input_size;
        
        // Find max deviation from average
        fp32 max_deviation = 0.0f;
        for (size_t i = 0; i < input_size; i++) {
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
        std::vector<i8> quantized_input(input_size);
        for (size_t i = 0; i < input_size; i++) {
            i32 temp = static_cast<i32>(std::round(Si * dataIn.get<fp32>(i))) + zi;
            // Clamp to int8 range
            quantized_input[i] = static_cast<i8>(std::max(-128, std::min(127, temp)));
        }

        // ===== STEP 5: Perform quantized convolution =====
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    // Start with quantized bias: bx = round(Sb * Bx)
                    i32 accumulator = static_cast<i32>(std::round(Sb * getBiasData().get<fp32>(m)));
                    
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
                                fp32 fp_weight = getWeightData().get<fp32>(weight_idx);
                                i32 temp_weight = static_cast<i32>(std::round(Sw * fp_weight));
                                i8 quantized_weight = static_cast<i8>(std::max(-128, std::min(127, temp_weight)));
                                
                                // int32 accumulation: ix * wx
                                accumulator += static_cast<i32>(quantized_input[input_idx]) * 
                                              static_cast<i32>(quantized_weight);
                            }
                        }
                    }
                    
                    // ===== STEP 6: Apply ReLU in quantized space =====
                    // ReLU: if accumulator < zi, set to zi (NOT zero!)
                    if (accumulator < static_cast<i32>(zi)) {
                        accumulator = static_cast<i32>(zi);
                    }
                    
                    // ===== STEP 7: Dequantize back to fp32 =====
                    // fp32_value = (int32_value - zi) / (Si * Sw)
                    fp32 dequantized = static_cast<fp32>(accumulator - zi) / (Si * Sw);
                    
                    // Store result
                    size_t output_idx = p * Q * M + q * M + m;
                    getOutputData().get<fp32>(output_idx) = dequantized;
                }
            }
        }
        
        // // Debug: Print first few output values
        // std::cout << "[DEBUG] First 3 quantized conv outputs: " 
        //           << getOutputData().get<fp32>(0) << ", " 
        //           << getOutputData().get<fp32>(1) << ", " 
        //           << getOutputData().get<fp32>(2) << std::endl;
    }
} // namespace ML

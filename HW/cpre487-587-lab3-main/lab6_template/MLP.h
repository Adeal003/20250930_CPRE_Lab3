#pragma once

#ifdef ZEDBOARD

#include <xil_io.h>
#include <xaxicdma_hw.h>
#include <xil_cache.h>
#include <xparameters.h>

using namespace ML;

#define MLP_BRAM_BASEADDR (0x40000000)
#define MLP_CONV_BASEADDR (0x4C000000)

#define MLP_INPUTS  ((ui8*)MLP_BRAM_BASEADDR + (0 << 17))
#define MLP_OUTPUTS ((ui8*)MLP_BRAM_BASEADDR + (1 << 17))
#define MLP_FILTER0 ((ui8*)MLP_BRAM_BASEADDR + (1 << 18) + (0 << 11))
#define MLP_FILTER1 ((ui8*)MLP_BRAM_BASEADDR + (1 << 18) + (1 << 11))
#define MLP_FILTER2 ((ui8*)MLP_BRAM_BASEADDR + (1 << 18) + (2 << 11))
#define MLP_FILTER3 ((ui8*)MLP_BRAM_BASEADDR + (1 << 18) + (3 << 11))

#define MLP_CTRLA                       (MLP_CONV_BASEADDR + 0x00)
#define MLP_CTRLA_CONV_IDLE             (1 << 0)
#define MLP_CTRLB                       (MLP_CONV_BASEADDR + 0x04)
#define MLP_CTRLB_SWAP_FILTERS          (1 << 0)
#define MLP_CTRLB_SWAP_ACTIVATIONS      (1 << 1)
#define MLP_CTRLB_MAX_POOLING           (1 << 2)
#define MLP_CTRLB_RELU                  (1 << 3)
// #define MLP_RESERVED                   (MLP_CONV_BASEADDR + 0x08)
#define MLP_FILTER_W                    (MLP_CONV_BASEADDR + 0x0C)
#define MLP_FILTER_H                    (MLP_CONV_BASEADDR + 0x10)
#define MLP_FILTER_C                    (MLP_CONV_BASEADDR + 0x14)
#define MLP_OUTPUT_W                    (MLP_CONV_BASEADDR + 0x18)
#define MLP_OUTPUT_H                    (MLP_CONV_BASEADDR + 0x1C)
#define MLP_INPUT_END_DIFF_FW           (MLP_CONV_BASEADDR + 0x20)
#define MLP_INPUT_END_DIFF_FH           (MLP_CONV_BASEADDR + 0x24)
#define MLP_INPUT_END_DIFF_FC           (MLP_CONV_BASEADDR + 0x28)
#define MLP_INPUT_END_DIFF_OW           (MLP_CONV_BASEADDR + 0x2C)
#define MLP_OUTPUT_ELEMENTS_PER_CHANNEL (MLP_CONV_BASEADDR + 0x30)
#define MLP_OUTPUT_INITIAL_OFFSET       (MLP_CONV_BASEADDR + 0x34)
#define MLP_MAC0_BIAS                   (MLP_CONV_BASEADDR + 0x38)
#define MLP_MAC1_BIAS                   (MLP_CONV_BASEADDR + 0x3C)
#define MLP_MAC2_BIAS                   (MLP_CONV_BASEADDR + 0x40)
#define MLP_MAC3_BIAS                   (MLP_CONV_BASEADDR + 0x44)
#define MLP_Q_SCALE                     (MLP_CONV_BASEADDR + 0x48)
#define MLP_Q_ZERO                      (MLP_CONV_BASEADDR + 0x4C)

static inline void memcpy_dma_start(void* dest, const void* src, ui32 len) {
    // std::cout << "MEMCPY FROM " << src << " TO " << dest << " OF LENGTH " << len << '\n';
    Xil_Out32(XPAR_AXI_CDMA_0_BASEADDR + XAXICDMA_DSTADDR_OFFSET, (ui32)dest);
    Xil_Out32(XPAR_AXI_CDMA_0_BASEADDR + XAXICDMA_SRCADDR_OFFSET, (ui32)src);
    Xil_Out32(XPAR_AXI_CDMA_0_BASEADDR + XAXICDMA_BTT_OFFSET, len);
    // volatile ui8* udest = (volatile ui8*)dest;
    // volatile ui8* usrc = (volatile ui8*)src;
    // volatile ui8* end = udest + len;
    // while (udest != end)
    //     *udest++ = *usrc++;
}

static inline bool memcpy_dma_idle() {
    return !!(Xil_In32(XPAR_AXI_CDMA_0_BASEADDR + XAXICDMA_SR_OFFSET) & XAXICDMA_SR_IDLE_MASK);
}

static inline void memcpy_dma(void* dest, const void* src, ui32 len) {
    memcpy_dma_start(dest, src, len);
    while (!memcpy_dma_idle());
}

static inline void memcheck_write(void* baseaddr, ui32 length, ui32 seed) {
    ui32* data = (ui32*)malloc(length);
    ui32 word = seed;
    for (ui32 i = 0; i < length/4; i++) {
        data[i] = word++;
    }
    memcpy_dma(baseaddr, data, length);
    free(data);
}

static inline bool memcheck_verify(void* baseaddr, ui32 length, ui32 seed) {
    ui32* data = (ui32*)malloc(length);
    memcpy_dma(data, baseaddr, length);
    ui32 word = seed;
    for (ui32 i = 0; i < length/4; i++) {
        ui32 expected = word++;
        ui32 actual = data[i];
        if (actual != expected) {
            std::cout << "Address " << std::hex << i << " incorrect. Expected: " << expected << " actual: " << actual << std::dec << '\n';
            free(data);
            return false;
        }
    }
    free(data);
    std::cout << "MEMCHECK of " << std::hex << baseaddr << std::dec << " complete!" << '\n';
    return true;
}

static inline void memcheck(void* dest, const void* src, ui32 len) {
    ui8* data = (ui8*)malloc(len);
    memcpy_dma(data, dest, len);
    
    for (ui32 i = 0; i < len; i++) {
        u8 expected = ((ui8*)src)[i];
        u8 actual = data[i];
        if (actual != expected) {
            std::cout << "Address " << std::hex << i << " incorrect. Expected: " << (int)expected << " actual: " << (int)actual << std::dec << '\n';
            free(data);
            return;
        }
    }
    std::cout << "MEMCHECK of " << std::hex << dest << std::dec << " complete!" << '\n';
}

static inline void runMemoryTest() {
    std::cout << "Starting DMA Transfer Test!\n";
    memcheck_write(MLP_INPUTS, (1 << 17), 14);
    memcheck_write(MLP_OUTPUTS, (1 << 17), 15);
    memcheck_verify(MLP_INPUTS, (1 << 17), 14);
    memcheck_verify(MLP_OUTPUTS, (1 << 17), 15);

    std::cout << "Write to bank 0\n";
    memcheck_write(MLP_FILTER0, (1 << 11), 10);
    memcheck_write(MLP_FILTER1, (1 << 11), 11);
    memcheck_write(MLP_FILTER2, (1 << 11), 12);
    memcheck_write(MLP_FILTER3, (1 << 11), 13);

    std::cout << "Write to bank 1\n";
    Xil_Out32(MLP_CTRLB, MLP_CTRLB_SWAP_FILTERS); // Swap filter banks
    memcheck_write(MLP_FILTER0, (1 << 11), 20);
    memcheck_write(MLP_FILTER1, (1 << 11), 21);
    memcheck_write(MLP_FILTER2, (1 << 11), 22);
    memcheck_write(MLP_FILTER3, (1 << 11), 23);
    memcheck_verify(MLP_INPUTS, (1 << 17), 14);
    memcheck_verify(MLP_OUTPUTS, (1 << 17), 15);

    std::cout << "Verify bank 1\n";
    memcheck_verify(MLP_FILTER0, (1 << 11), 20);
    memcheck_verify(MLP_FILTER1, (1 << 11), 21);
    memcheck_verify(MLP_FILTER2, (1 << 11), 22);
    memcheck_verify(MLP_FILTER3, (1 << 11), 23);

    std::cout << "Verify bank 0\n";
    Xil_Out32(MLP_CTRLB, MLP_CTRLB_SWAP_ACTIVATIONS); // Swap banks back
    memcheck_verify(MLP_FILTER0, (1 << 11), 10);
    memcheck_verify(MLP_FILTER1, (1 << 11), 11);
    memcheck_verify(MLP_FILTER2, (1 << 11), 12);
    memcheck_verify(MLP_FILTER3, (1 << 11), 13);
    memcheck_verify(MLP_INPUTS, (1 << 17), 15);
    memcheck_verify(MLP_OUTPUTS, (1 << 17), 14);
}

#endif

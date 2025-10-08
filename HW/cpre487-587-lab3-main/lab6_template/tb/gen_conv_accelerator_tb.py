################################################################
# Accelerator Testbench Generator
# See the middle of the file for the convolutions being tested
#
# Gregory Ling, 2024
################################################################

import sys
import numpy as np

BRAM_SIZE_BYTES = 128
control_process = ''

index_gen_input_addr = []
index_gen_filter_addr = []
index_gen_tlast = []
mac0_in_tdata = []
mac0_in_tlast = []
mac1_in_tdata = []
mac1_in_tlast = []
mac2_in_tdata = []
mac2_in_tlast = []
mac3_in_tdata = []
mac3_in_tlast = []
mac0_out_tdata = []
mac0_out_tlast = []
mac1_out_tdata = []
mac1_out_tlast = []
mac2_out_tdata = []
mac2_out_tlast = []
mac3_out_tdata = []
mac3_out_tlast = []
combined_out_tdata = []
combined_out_tlast = []
combined_out_tid = []
deq_out_tdata = []
deq_out_tlast = []
deq_out_tid = []
bram_output_write_addr = []
bram_output_write_data = []


def indent(str, tabs):
    return str.replace('\n', '\n' + ('    ' * tabs))

def u32_v(value):
    return f'x"{value:08X}"'
    
def convolve(input, filter, biases, scale, zero, max_pooling, relu, output_initial_offset):
    global control_process
    # Create static BRAM data vectors
    flat_input = np.int8(input).flatten()
    flat_filter0 = np.int8(filter[0]).flatten()
    flat_filter1 = np.int8(filter[1]).flatten()
    flat_filter2 = np.int8(filter[2]).flatten()
    flat_filter3 = np.int8(filter[3]).flatten()
    IW = np.shape(input)[2]
    IH = np.shape(input)[1]
    FC = np.shape(filter)[1]
    FH = np.shape(filter)[2]
    FW = np.shape(filter)[3]
    OW = IW - FW + 1
    OH = IH - FH + 1
    input_end_diff_fw = 1 - FW + IW
    input_end_diff_fh = input_end_diff_fw - (IW*FH) + (IW*IH)
    input_end_diff_fc = input_end_diff_fh - (IW * IH * FC) + 1
    input_end_diff_ow = input_end_diff_fc + (FW - 1)
    output_elements_per_channel = int((OW * OH)/4) if max_pooling else OW * OH

    control_process += f"""\
conv_idle <= '1';
wait for 10ps;
BRAM_INPUT_data <= x"{('A5'*(BRAM_SIZE_BYTES-len(flat_input)))}{''.join(f'{np.uint8(x):02X}' for x in reversed(flat_input))}";
BRAM_FILTER0_data <= x"{('A5'*(BRAM_SIZE_BYTES-len(flat_filter0)))}{''.join(f'{np.uint8(x):02X}' for x in reversed(flat_filter0))}";
BRAM_FILTER1_data <= x"{('A5'*(BRAM_SIZE_BYTES-len(flat_filter1)))}{''.join(f'{np.uint8(x):02X}' for x in reversed(flat_filter1))}";
BRAM_FILTER2_data <= x"{('A5'*(BRAM_SIZE_BYTES-len(flat_filter2)))}{''.join(f'{np.uint8(x):02X}' for x in reversed(flat_filter2))}";
BRAM_FILTER3_data <= x"{('A5'*(BRAM_SIZE_BYTES-len(flat_filter3)))}{''.join(f'{np.uint8(x):02X}' for x in reversed(flat_filter3))}";
max_pooling <= '{int(max_pooling)}';
relu <= '{int(relu)}';
filter_w <= x"{np.shape(filter)[3]:08X}";
filter_h <= x"{np.shape(filter)[2]:08X}";
filter_c <= x"{np.shape(filter)[1]:08X}";
output_w <= x"{np.shape(input)[2] - np.shape(filter)[3] + 1:08X}";
output_h <= x"{np.shape(input)[1] - np.shape(filter)[2] + 1:08X}";
input_end_diff_fw <= x"{np.uint32(input_end_diff_fw):08X}";
input_end_diff_fh <= x"{np.uint32(input_end_diff_fh):08X}";
input_end_diff_fc <= x"{np.uint32(input_end_diff_fc):08X}";
input_end_diff_ow <= x"{np.uint32(input_end_diff_ow):08X}";
output_elements_per_channel <= x"{output_elements_per_channel:08X}";
output_initial_offset <= x"{output_initial_offset:08X}";
mac0_bias <= x"{np.uint32(biases[0]):08X}";
mac1_bias <= x"{np.uint32(biases[1]):08X}";
mac2_bias <= x"{np.uint32(biases[2]):08X}";
mac3_bias <= x"{np.uint32(biases[3]):08X}";
q_scale <= x"{scale:08X}";
q_zero <= x"{np.uint32(zero):08X}";
wait for 10ps;
conv_idle <= '0';
wait until rising_edge(conv_complete);
wait for 10ps;
"""

    output_buffer = np.zeros((4, int(OH/2) if max_pooling else OH, int(OW/2) if max_pooling else OW), dtype=np.int8)

    for oh in range(OH):
        for ow in range(OW):
            output_addr = ((int(oh/2)*int(OW/2)) + int(ow/2)) if max_pooling else ((oh * OW) + (ow))
            out0 = np.int32(biases[0])
            out1 = np.int32(biases[1])
            out2 = np.int32(biases[2])
            out3 = np.int32(biases[3])
            for fc in range(FC):
                for fh in range(FH):
                    for fw in range(FW):
                        input_addr = (fc * IH * IW) + ((fh+oh) * IW) + (fw+ow)
                        filter_addr = (fc * FH * FW) + ((fh) * FW) + (fw)
                        last = (fc == FC-1) and (fh == FH-1) and (fw == FW-1)
                        index_gen_input_addr.append(input_addr)
                        index_gen_filter_addr.append(filter_addr)
                        index_gen_tlast.append(last)
                        input_val = input[fc][fh+oh][fw+ow]
                        filter0_val = filter[0][fc][fh][fw]
                        filter1_val = filter[1][fc][fh][fw]
                        filter2_val = filter[2][fc][fh][fw]
                        filter3_val = filter[3][fc][fh][fw]
                        mac0_in_tdata.append((np.uint8(input_val) << 8) | np.uint8(filter0_val))
                        mac1_in_tdata.append((np.uint8(input_val) << 8) | np.uint8(filter1_val))
                        mac2_in_tdata.append((np.uint8(input_val) << 8) | np.uint8(filter2_val))
                        mac3_in_tdata.append((np.uint8(input_val) << 8) | np.uint8(filter3_val))
                        mac0_in_tlast.append(last)
                        mac1_in_tlast.append(last)
                        mac2_in_tlast.append(last)
                        mac3_in_tlast.append(last)
                        out0 += input_val * filter0_val
                        out1 += input_val * filter1_val
                        out2 += input_val * filter2_val
                        out3 += input_val * filter3_val

            mac0_out_tdata.append(out0)
            mac1_out_tdata.append(out1)
            mac2_out_tdata.append(out2)
            mac3_out_tdata.append(out3)
            mac0_out_tlast.append(True)
            mac1_out_tlast.append(True)
            mac2_out_tlast.append(True)
            mac3_out_tlast.append(True)
            for i, out in enumerate((out0, out1, out2, out3)):
                combined_out_tdata.append(out)
                combined_out_tlast.append(True)
                combined_out_tid.append(i)
                scaled = (out * scale) >> 32
                relued = max(scaled, 0) if relu else scaled
                zeroed = relued + zero
                saturated = max(min(zeroed, 127), -128)
                deq_out_tdata.append(saturated)
                deq_out_tlast.append(True)
                deq_out_tid.append(i)

                bram_output_write_addr.append(4 * int((output_initial_offset + output_addr + (output_elements_per_channel*i))/4))
                if max_pooling:
                    do_max = (oh % 2 == 1 or ow % 2 == 1)
                    maxed = max(saturated, output_buffer[i][int(oh/2)][int(ow/2)]) if do_max else saturated
                    output_buffer[i][int(oh/2)][int(ow/2)] = maxed
                else:
                    output_buffer[i][oh][ow] = saturated
                
                bram_output_write_data.append(output_buffer.flatten().view(np.uint32)[int((output_addr + (output_elements_per_channel*i))/4)])
    print(output_buffer, file=sys.stderr)

inputs = np.array([
    [
        [127, -1, -128, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 0],
        [21, 3, 2, 1],
    ]
])

filters = np.array([
    [
        [
            [127, -1, -128],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
    ],
    [
        [
            [-13, -14, -15],
            [-16, -17, -18],
        ],
        [
            [-19, -20, -21],
            [-22, -23, -24],
        ],
    ],
    [
        [
            [25, 26, 27],
            [28, 29, 30],
        ],
        [
            [31, 32, 33],
            [34, 35, 36],
        ],
    ],
    [
        [
            [37, 38, 39],
            [40, 41, 42],
        ],
        [
            [43, 44, 45],
            [46, 47, 48],
        ],
    ]
])


convolve(inputs, filters, [0, 1, 2, 3], 0x4000000, 0, False, False, 0)
convolve(inputs, filters, [0, 1, -2, 0x8A32BC81], 0x7A32BC81, -127, False, False, 0)
convolve(np.reshape(range(-20, 20), (2, 5, 4)), filters, [0, 1, 2, 3], 0x7A32BC81, -127, True, True, 0)
convolve(np.reshape(range(-30, 30), (2, 5, 6)), filters, [4, 5, 6, 7], 0x7A32BC81, -100, True, True, 4)

inputs = [[[0x40, 0], [0, 0]]]
filters = [[[[1, 0], [0, 0]]], [[[1, 0], [0, 0]]], [[[1, 0], [0, 0]]], [[[1, 0], [0, 0]]]]

convolve(inputs, filters, [0x100, 0x100, 0x100, 0x100], 0x40000000, 3, False, True, 0)


def vhdl_type(bits):
    return "std_logic" if bits == 1 else f"std_logic_vector({bits-1} downto 0)"

def external_signal(name, bits):
    return f'<< signal dut.{name} : {vhdl_type(bits)} >>'

def literal(value, bits):
    if bits == 1:
        return f"'{int(value)}'"
    else:
        return f'"{int(value):0{bits}b}"'
    

def gen_axis_checking_process(prefix, signals):
    out = "process\n"
    out += "    variable i : integer := 0;\n"
    num_values = len(signals[0][1])
    for name, values, bits in signals:
        bitstring = ''.join(f"{np.uint32(v):0{bits}b}"[-bits:] for v in reversed(values))
        out += f'    constant EXPECTED_VALUES_{name} : std_logic_vector({num_values*bits}-1 downto 0) := "{bitstring}";\n'
    out += "begin\n"
    for name, values, bits in signals:
        if bits == 1:
            out += f'    EXPECTED_{name} <= EXPECTED_VALUES_{name}(i*{bits});\n'
        else:
            out += f'    EXPECTED_{name} <= EXPECTED_VALUES_{name}((i+1)*{bits}-1 downto i*{bits});\n'
    out += f"    wait until rising_edge(clk) and TEST_{prefix}_tready = '1' and TEST_{prefix}_tvalid = '1';\n"
    for name, values, bits in signals:
        out += f'    assert TEST_{name} = EXPECTED_{name} report "ASSERTION FAILURE";\n'
        out += f"    if not (TEST_{name} = EXPECTED_{name}) then TEST_{prefix}_fail <= 'X'; end if;\n"
    out += "    i := i + 1;\n"
    out += f"    if (i = {num_values}) then\n"
    out += f"        wait until rising_edge(clk) and TEST_{prefix}_tready = '1' and TEST_{prefix}_tvalid = '1';\n"
    out += f"        TEST_{prefix}_fail <= 'X';\n"
    out += f'        assert FALSE report "TOO MANY TRANSACTIONS!!!";\n'
    out += f'    end if;\n'
    out += "end process;\n"
    return out

def gen_bram_checking_process(prefix, signals):
    out = "process\n"
    out += "    variable i : integer := 0;\n"
    num_values = len(signals[0][1])
    for name, values, bits in signals:
        bitstring = ''.join(f"{np.uint32(v):0{bits}b}"[-bits:] for v in reversed(values))
        out += f'    constant EXPECTED_VALUES_{name} : std_logic_vector({num_values*bits}-1 downto 0) := "{bitstring}";\n'
    out += "begin\n"
    for name, values, bits in signals:
        if bits == 1:
            out += f'    EXPECTED_{name} <= EXPECTED_VALUES_{name}(i*{bits});\n'
        else:
            out += f'    EXPECTED_{name} <= EXPECTED_VALUES_{name}((i+1)*{bits}-1 downto i*{bits});\n'
    out += f"    wait until rising_edge(clk) and {prefix}_en = '1' and {prefix}_we = \"1111\";\n"
    for name, values, bits in signals:
        out += f'    assert {name} = EXPECTED_{name} report "ASSERTION FAILURE";\n'
        out += f"    if not ({name} = EXPECTED_{name}) then {prefix}_fail <= 'X'; end if;\n"
    out += "    i := i + 1;\n"
    out += f"    if (i = {num_values}) then\n"
    out += f"        wait until rising_edge(clk) and {prefix}_en = '1' and {prefix}_we = \"1111\";\n"
    out += f"        {prefix}_fail <= 'X';\n"
    out += f'        assert FALSE report "TOO MANY TRANSACTIONS!!!";\n'
    out += f'    end if;\n'
    out += "end process;\n"
    return out

print(f"""\
----------------------------------------------------------------------------------
-- AUTOGENERATED. See gen_conv_accelerator_tb.py
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity conv_accelerator_tb is
end conv_accelerator_tb;

architecture Behavioral of conv_accelerator_tb is

    constant DIM_WIDTH : integer := 12; -- Max dim size is 2048 in a dense layer
    constant INPUT_ADDR_WIDTH : integer := 7; -- Max input size is 60*60*32 < 2^17
    constant FILTER_ADDR_WIDTH : integer := 7; -- Max filter size is 5*5*32 < 2^10
    constant OUTPUT_ADDR_WIDTH : integer := 7; -- Max size of output channel is 60x60, 4 channels < 2^14
    constant INPUT_BRAM_ADDR_WIDTH : integer := 5; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
    constant FILTER_BRAM_ADDR_WIDTH : integer := 5; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
    constant OUTPUT_BRAM_ADDR_WIDTH : integer := 5; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
    constant BRAM_DATA_WIDTH : integer := 32; -- Data width of raw BRAM interface
    constant MAC_DATA_WIDTH : integer := 8; -- Data width of each MAC input operand, defaults to int8. Supports sub-byte indexing, must be power of 2 and less than BRAM_DATA_WIDTH
    constant MAC_OUTPUT_DATA_WIDTH : integer := 32; -- Data width of the raw output of the MAC unit

    -- Configuration values from conv_config unit
    signal max_pooling : std_logic;
    signal relu : std_logic;
    signal filter_w : std_logic_vector(31 downto 0);
    signal filter_h : std_logic_vector(31 downto 0);
    signal filter_c : std_logic_vector(31 downto 0);
    signal output_w : std_logic_vector(31 downto 0);
    signal output_h : std_logic_vector(31 downto 0);
    signal input_end_diff_fw : std_logic_vector(31 downto 0);
    signal input_end_diff_fh : std_logic_vector(31 downto 0);
    signal input_end_diff_fc : std_logic_vector(31 downto 0);
    signal input_end_diff_ow : std_logic_vector(31 downto 0);
    signal output_elements_per_channel : std_logic_vector(31 downto 0);
    signal output_initial_offset : std_logic_vector(31 downto 0);
    signal mac0_bias : std_logic_vector(31 downto 0);
    signal mac1_bias : std_logic_vector(31 downto 0);
    signal mac2_bias : std_logic_vector(31 downto 0);
    signal mac3_bias : std_logic_vector(31 downto 0);
    signal q_scale : std_logic_vector(31 downto 0);
    signal q_zero : std_logic_vector(31 downto 0);

    -- BRAM blocks for high speed memory access
    signal BRAM_INPUT_addr : std_logic_vector(32-1 downto 0);
    signal BRAM_INPUT_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal BRAM_INPUT_dout : std_logic_vector(31 downto 0);
    signal BRAM_INPUT_dout_delay1 : std_logic_vector(31 downto 0);
    signal BRAM_INPUT_en : std_logic;
    signal BRAM_INPUT_we : std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
    signal BRAM_INPUT_rst : std_logic;
    signal BRAM_INPUT_clk : std_logic;

    signal BRAM_FILTER0_addr : std_logic_vector(32-1 downto 0);
    signal BRAM_FILTER0_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal BRAM_FILTER0_dout : std_logic_vector(31 downto 0);
    signal BRAM_FILTER0_dout_delay1 : std_logic_vector(31 downto 0);
    signal BRAM_FILTER0_en : std_logic;
    signal BRAM_FILTER0_we : std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
    signal BRAM_FILTER0_rst : std_logic;
    signal BRAM_FILTER0_clk : std_logic;

    signal BRAM_FILTER1_addr : std_logic_vector(32-1 downto 0);
    signal BRAM_FILTER1_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal BRAM_FILTER1_dout : std_logic_vector(31 downto 0);
    signal BRAM_FILTER1_dout_delay1 : std_logic_vector(31 downto 0);
    signal BRAM_FILTER1_en : std_logic;
    signal BRAM_FILTER1_we : std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
    signal BRAM_FILTER1_rst : std_logic;
    signal BRAM_FILTER1_clk : std_logic;

    signal BRAM_FILTER2_addr : std_logic_vector(32-1 downto 0);
    signal BRAM_FILTER2_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal BRAM_FILTER2_dout : std_logic_vector(31 downto 0);
    signal BRAM_FILTER2_dout_delay1 : std_logic_vector(31 downto 0);
    signal BRAM_FILTER2_en : std_logic;
    signal BRAM_FILTER2_we : std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
    signal BRAM_FILTER2_rst : std_logic;
    signal BRAM_FILTER2_clk : std_logic;

    signal BRAM_FILTER3_addr : std_logic_vector(32-1 downto 0);
    signal BRAM_FILTER3_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal BRAM_FILTER3_dout : std_logic_vector(31 downto 0);
    signal BRAM_FILTER3_dout_delay1 : std_logic_vector(31 downto 0);
    signal BRAM_FILTER3_en : std_logic;
    signal BRAM_FILTER3_we : std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
    signal BRAM_FILTER3_rst : std_logic;
    signal BRAM_FILTER3_clk : std_logic;

    signal BRAM_OUTPUT_addr : std_logic_vector(32-1 downto 0);
    signal EXPECTED_BRAM_OUTPUT_addr : std_logic_vector(32-1 downto 0);
    signal BRAM_OUTPUT_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal EXPECTED_BRAM_OUTPUT_din : std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
    signal BRAM_OUTPUT_dout : std_logic_vector(31 downto 0);
    signal BRAM_OUTPUT_dout_delay1 : std_logic_vector(31 downto 0);
    signal BRAM_OUTPUT_en : std_logic;
    signal BRAM_OUTPUT_we : std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
    signal BRAM_OUTPUT_rst : std_logic;
    signal BRAM_OUTPUT_clk : std_logic;
    signal BRAM_OUTPUT_fail : std_logic := '0';

    signal BRAM_INPUT_data : std_logic_vector(8*{BRAM_SIZE_BYTES}-1 downto 0);
    signal BRAM_FILTER0_data : std_logic_vector(8*{BRAM_SIZE_BYTES}-1 downto 0);
    signal BRAM_FILTER1_data : std_logic_vector(8*{BRAM_SIZE_BYTES}-1 downto 0);
    signal BRAM_FILTER2_data : std_logic_vector(8*{BRAM_SIZE_BYTES}-1 downto 0);
    signal BRAM_FILTER3_data : std_logic_vector(8*{BRAM_SIZE_BYTES}-1 downto 0);
    signal BRAM_OUTPUT_data : std_logic_vector(8*{BRAM_SIZE_BYTES}-1 downto 0);
    
    signal conv_complete : std_logic; -- Reset the convolutional logic, must be set between each convolutional operation
    signal conv_idle : std_logic; -- Reset the convolutional logic, must be set between each convolutional operation
    signal rst : std_logic; -- Reset everything, including BRAM contents
    signal clk : std_logic := '0';
    
    signal TEST_s_index_gen_m_axis_tready : std_logic;
    signal TEST_s_index_gen_m_axis_tdata_input_addr : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal EXPECTED_s_index_gen_m_axis_tdata_input_addr : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal TEST_s_index_gen_m_axis_tdata_filter_addr : std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
    signal EXPECTED_s_index_gen_m_axis_tdata_filter_addr : std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
    signal TEST_s_index_gen_m_axis_tlast : std_logic;
    signal EXPECTED_s_index_gen_m_axis_tlast : std_logic;
    signal TEST_s_index_gen_m_axis_tvalid : std_logic;    
    signal TEST_s_index_gen_m_axis_fail : std_logic := '0';

    signal TEST_s_mac0_s_axis_tready : std_logic;
    signal TEST_s_mac0_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal EXPECTED_s_mac0_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal TEST_s_mac0_s_axis_tlast : std_logic;
    signal EXPECTED_s_mac0_s_axis_tlast : std_logic;
    signal TEST_s_mac0_s_axis_tvalid : std_logic;
    signal TEST_s_mac0_s_axis_fail : std_logic := '0';
    signal TEST_s_mac1_s_axis_tready : std_logic;
    signal TEST_s_mac1_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal EXPECTED_s_mac1_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal TEST_s_mac1_s_axis_tlast : std_logic;
    signal EXPECTED_s_mac1_s_axis_tlast : std_logic;
    signal TEST_s_mac1_s_axis_tvalid : std_logic;
    signal TEST_s_mac1_s_axis_fail : std_logic := '0';
    signal TEST_s_mac2_s_axis_tready : std_logic;
    signal TEST_s_mac2_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal EXPECTED_s_mac2_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal TEST_s_mac2_s_axis_tlast : std_logic;
    signal EXPECTED_s_mac2_s_axis_tlast : std_logic;
    signal TEST_s_mac2_s_axis_tvalid : std_logic;
    signal TEST_s_mac2_s_axis_fail : std_logic := '0';
    signal TEST_s_mac3_s_axis_tready : std_logic;
    signal TEST_s_mac3_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal EXPECTED_s_mac3_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal TEST_s_mac3_s_axis_tlast : std_logic;
    signal EXPECTED_s_mac3_s_axis_tlast : std_logic;
    signal TEST_s_mac3_s_axis_tvalid : std_logic;
    signal TEST_s_mac3_s_axis_fail : std_logic := '0';

    signal TEST_s_mac0_m_axis_tready : std_logic;
    signal TEST_s_mac0_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal EXPECTED_s_mac0_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal TEST_s_mac0_m_axis_tlast : std_logic;
    signal EXPECTED_s_mac0_m_axis_tlast : std_logic;
    signal TEST_s_mac0_m_axis_tvalid : std_logic;
    signal TEST_s_mac0_m_axis_fail : std_logic := '0';
    signal TEST_s_mac1_m_axis_tready : std_logic;
    signal TEST_s_mac1_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal EXPECTED_s_mac1_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal TEST_s_mac1_m_axis_tlast : std_logic;
    signal EXPECTED_s_mac1_m_axis_tlast : std_logic;
    signal TEST_s_mac1_m_axis_tvalid : std_logic;
    signal TEST_s_mac1_m_axis_fail : std_logic := '0';
    signal TEST_s_mac2_m_axis_tready : std_logic;
    signal TEST_s_mac2_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal EXPECTED_s_mac2_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal TEST_s_mac2_m_axis_tlast : std_logic;
    signal EXPECTED_s_mac2_m_axis_tlast : std_logic;
    signal TEST_s_mac2_m_axis_tvalid : std_logic;
    signal TEST_s_mac2_m_axis_fail : std_logic := '0';
    signal TEST_s_mac3_m_axis_tready : std_logic;
    signal TEST_s_mac3_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal EXPECTED_s_mac3_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal TEST_s_mac3_m_axis_tlast : std_logic;
    signal EXPECTED_s_mac3_m_axis_tlast : std_logic;
    signal TEST_s_mac3_m_axis_tvalid : std_logic;
    signal TEST_s_mac3_m_axis_fail : std_logic := '0';

    signal TEST_s_out_combiner_m_axis_tready : std_logic;
    signal TEST_s_out_combiner_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal EXPECTED_s_out_combiner_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal TEST_s_out_combiner_m_axis_tlast : std_logic;
    signal EXPECTED_s_out_combiner_m_axis_tlast : std_logic;
    signal TEST_s_out_combiner_m_axis_tid : std_logic_vector(1 downto 0);
    signal EXPECTED_s_out_combiner_m_axis_tid : std_logic_vector(1 downto 0);
    signal TEST_s_out_combiner_m_axis_tvalid : std_logic;
    signal TEST_s_out_combiner_m_axis_fail : std_logic := '0';

    signal TEST_s_dequantization_m_axis_tready : std_logic;
    signal TEST_s_dequantization_m_axis_tdata : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal EXPECTED_s_dequantization_m_axis_tdata : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal TEST_s_dequantization_m_axis_tlast : std_logic;
    signal EXPECTED_s_dequantization_m_axis_tlast : std_logic;
    signal TEST_s_dequantization_m_axis_tid : std_logic_vector(1 downto 0);
    signal EXPECTED_s_dequantization_m_axis_tid : std_logic_vector(1 downto 0);
    signal TEST_s_dequantization_m_axis_tvalid : std_logic;
    signal TEST_s_dequantization_m_axis_fail : std_logic := '0';
begin
        
    BRAM_INPUT_dout <= BRAM_INPUT_dout_delay1; -- BRAM read latency = 2
    process(BRAM_INPUT_clk)
    begin
        if rising_edge(BRAM_INPUT_clk) then
            if (BRAM_INPUT_rst = '1') then
                BRAM_INPUT_dout_delay1 <= (others => '0');
            elsif (BRAM_INPUT_en = '1') then
                BRAM_INPUT_dout_delay1 <= BRAM_INPUT_data((32*(to_integer(unsigned(BRAM_INPUT_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_INPUT_addr(32-1 downto 2))))));
            end if;
        end if;
    end process;

    BRAM_FILTER0_dout <= BRAM_FILTER0_dout_delay1; -- BRAM read latency = 2
    process(BRAM_FILTER0_clk)
    begin
        if rising_edge(BRAM_FILTER0_clk) then
            if (BRAM_FILTER0_rst = '1') then
                BRAM_FILTER0_dout_delay1 <= (others => '0');
            elsif (BRAM_FILTER0_en = '1') then
                BRAM_FILTER0_dout_delay1 <= BRAM_FILTER0_data((32*(to_integer(unsigned(BRAM_FILTER0_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_FILTER0_addr(32-1 downto 2))))));
            end if;
        end if;
    end process;

    BRAM_FILTER1_dout <= BRAM_FILTER1_dout_delay1; -- BRAM read latency = 2
    process(BRAM_FILTER1_clk)
    begin
        if rising_edge(BRAM_FILTER1_clk) then
            if (BRAM_FILTER1_rst = '1') then
                BRAM_FILTER1_dout_delay1 <= (others => '0');
            elsif (BRAM_FILTER1_en = '1') then
                BRAM_FILTER1_dout_delay1 <= BRAM_FILTER1_data((32*(to_integer(unsigned(BRAM_FILTER1_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_FILTER1_addr(32-1 downto 2))))));
            end if;
        end if;
    end process;

    BRAM_FILTER2_dout <= BRAM_FILTER2_dout_delay1; -- BRAM read latency = 2
    process(BRAM_FILTER2_clk)
    begin
        if rising_edge(BRAM_FILTER2_clk) then
            if (BRAM_FILTER2_rst = '1') then
                BRAM_FILTER2_dout_delay1 <= (others => '0');
            elsif (BRAM_FILTER2_en = '1') then
                BRAM_FILTER2_dout_delay1 <= BRAM_FILTER2_data((32*(to_integer(unsigned(BRAM_FILTER2_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_FILTER2_addr(32-1 downto 2))))));
            end if;
        end if;
    end process;

    BRAM_FILTER3_dout <= BRAM_FILTER3_dout_delay1; -- BRAM read latency = 2
    process(BRAM_FILTER3_clk)
    begin
        if rising_edge(BRAM_FILTER3_clk) then
            if (BRAM_FILTER3_rst = '1') then
                BRAM_FILTER3_dout_delay1 <= (others => '0');
            elsif (BRAM_FILTER3_en = '1') then
                BRAM_FILTER3_dout_delay1 <= BRAM_FILTER3_data((32*(to_integer(unsigned(BRAM_FILTER3_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_FILTER3_addr(32-1 downto 2))))));
            end if;
        end if;
    end process;
       
    BRAM_OUTPUT_dout <= BRAM_OUTPUT_dout_delay1; -- BRAM read latency = 2
    process(BRAM_OUTPUT_clk)
    begin
        if rising_edge(BRAM_OUTPUT_clk) then
            if (BRAM_OUTPUT_rst = '1' or conv_idle = '1') then
                BRAM_OUTPUT_dout_delay1 <= (others => '0');
                BRAM_OUTPUT_data <= (others => '0');
            elsif (BRAM_OUTPUT_en = '1') then
                if (BRAM_OUTPUT_we = "1111") then
                    BRAM_OUTPUT_data((32*(to_integer(unsigned(BRAM_OUTPUT_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_OUTPUT_addr(32-1 downto 2)))))) <= BRAM_OUTPUT_din;
                    BRAM_OUTPUT_dout_delay1 <= BRAM_OUTPUT_din;
                else
                    BRAM_OUTPUT_dout_delay1 <= BRAM_OUTPUT_data((32*(to_integer(unsigned(BRAM_OUTPUT_addr(32-1 downto 2)))+1))-1 downto (32*(to_integer(unsigned(BRAM_OUTPUT_addr(32-1 downto 2))))));
                end if;
            end if;
        end if;
    end process;
 

    clk <= not clk after 1ps;

    dut: entity work.conv_accelerator
        generic map(
            DIM_WIDTH => DIM_WIDTH,
            INPUT_ADDR_WIDTH => INPUT_ADDR_WIDTH,
            FILTER_ADDR_WIDTH => FILTER_ADDR_WIDTH,
            OUTPUT_ADDR_WIDTH => OUTPUT_ADDR_WIDTH,
            INPUT_BRAM_ADDR_WIDTH => INPUT_BRAM_ADDR_WIDTH,
            FILTER_BRAM_ADDR_WIDTH => FILTER_BRAM_ADDR_WIDTH,
            OUTPUT_BRAM_ADDR_WIDTH => OUTPUT_BRAM_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            MAC_DATA_WIDTH => MAC_DATA_WIDTH,
            MAC_OUTPUT_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH
        )
        port map(
            -- Configuration values from conv_config unit
            max_pooling => max_pooling,
            relu => relu,
            filter_w => filter_w(DIM_WIDTH-1 downto 0),
            filter_h => filter_h(DIM_WIDTH-1 downto 0),
            filter_c => filter_c(DIM_WIDTH-1 downto 0),
            output_w => output_w(DIM_WIDTH-1 downto 0),
            output_h => output_h(DIM_WIDTH-1 downto 0),
            input_end_diff_fw => input_end_diff_fw(INPUT_ADDR_WIDTH-1 downto 0),
            input_end_diff_fh => input_end_diff_fh(INPUT_ADDR_WIDTH-1 downto 0),
            input_end_diff_fc => input_end_diff_fc(INPUT_ADDR_WIDTH-1 downto 0),
            input_end_diff_ow => input_end_diff_ow(INPUT_ADDR_WIDTH-1 downto 0),
            output_elements_per_channel => output_elements_per_channel(OUTPUT_ADDR_WIDTH-1 downto 0),
            output_initial_offset => output_initial_offset(OUTPUT_ADDR_WIDTH-1 downto 0),
            mac0_bias => mac0_bias(MAC_OUTPUT_DATA_WIDTH-1 downto 0),
            mac1_bias => mac1_bias(MAC_OUTPUT_DATA_WIDTH-1 downto 0),
            mac2_bias => mac2_bias(MAC_OUTPUT_DATA_WIDTH-1 downto 0),
            mac3_bias => mac3_bias(MAC_OUTPUT_DATA_WIDTH-1 downto 0),
            q_scale => q_scale(MAC_OUTPUT_DATA_WIDTH-1 downto 0),
            q_zero => q_zero(MAC_DATA_WIDTH-1 downto 0),

            -- BRAM blocks for high speed memory access
            BRAM_INPUT_addr => BRAM_INPUT_addr,
            BRAM_INPUT_din => BRAM_INPUT_din,
            BRAM_INPUT_dout => BRAM_INPUT_dout(BRAM_DATA_WIDTH-1 downto 0),
            BRAM_INPUT_en => BRAM_INPUT_en,
            BRAM_INPUT_we => BRAM_INPUT_we,
            BRAM_INPUT_rst => BRAM_INPUT_rst,
            BRAM_INPUT_clk => BRAM_INPUT_clk,

            BRAM_FILTER0_addr => BRAM_FILTER0_addr,
            BRAM_FILTER0_din => BRAM_FILTER0_din,
            BRAM_FILTER0_dout => BRAM_FILTER0_dout(BRAM_DATA_WIDTH-1 downto 0),
            BRAM_FILTER0_en => BRAM_FILTER0_en,
            BRAM_FILTER0_we => BRAM_FILTER0_we,
            BRAM_FILTER0_rst => BRAM_FILTER0_rst,
            BRAM_FILTER0_clk => BRAM_FILTER0_clk,

            BRAM_FILTER1_addr => BRAM_FILTER1_addr,
            BRAM_FILTER1_din => BRAM_FILTER1_din,
            BRAM_FILTER1_dout => BRAM_FILTER1_dout(BRAM_DATA_WIDTH-1 downto 0),
            BRAM_FILTER1_en => BRAM_FILTER1_en,
            BRAM_FILTER1_we => BRAM_FILTER1_we,
            BRAM_FILTER1_rst => BRAM_FILTER1_rst,
            BRAM_FILTER1_clk => BRAM_FILTER1_clk,

            BRAM_FILTER2_addr => BRAM_FILTER2_addr,
            BRAM_FILTER2_din => BRAM_FILTER2_din,
            BRAM_FILTER2_dout => BRAM_FILTER2_dout(BRAM_DATA_WIDTH-1 downto 0),
            BRAM_FILTER2_en => BRAM_FILTER2_en,
            BRAM_FILTER2_we => BRAM_FILTER2_we,
            BRAM_FILTER2_rst => BRAM_FILTER2_rst,
            BRAM_FILTER2_clk => BRAM_FILTER2_clk,

            BRAM_FILTER3_addr => BRAM_FILTER3_addr,
            BRAM_FILTER3_din => BRAM_FILTER3_din,
            BRAM_FILTER3_dout => BRAM_FILTER3_dout(BRAM_DATA_WIDTH-1 downto 0),
            BRAM_FILTER3_en => BRAM_FILTER3_en,
            BRAM_FILTER3_we => BRAM_FILTER3_we,
            BRAM_FILTER3_rst => BRAM_FILTER3_rst,
            BRAM_FILTER3_clk => BRAM_FILTER3_clk,

            BRAM_OUTPUT_addr => BRAM_OUTPUT_addr,
            BRAM_OUTPUT_din => BRAM_OUTPUT_din,
            BRAM_OUTPUT_dout => BRAM_OUTPUT_dout(BRAM_DATA_WIDTH-1 downto 0),
            BRAM_OUTPUT_en => BRAM_OUTPUT_en,
            BRAM_OUTPUT_we => BRAM_OUTPUT_we,
            BRAM_OUTPUT_rst => BRAM_OUTPUT_rst,
            BRAM_OUTPUT_clk => BRAM_OUTPUT_clk,

            TEST_s_index_gen_m_axis_tready => TEST_s_index_gen_m_axis_tready,
            TEST_s_index_gen_m_axis_tdata_input_addr => TEST_s_index_gen_m_axis_tdata_input_addr,
            TEST_s_index_gen_m_axis_tdata_filter_addr => TEST_s_index_gen_m_axis_tdata_filter_addr,
            TEST_s_index_gen_m_axis_tlast => TEST_s_index_gen_m_axis_tlast,
            TEST_s_index_gen_m_axis_tvalid => TEST_s_index_gen_m_axis_tvalid,

            TEST_s_mac0_s_axis_tready => TEST_s_mac0_s_axis_tready,
            TEST_s_mac0_s_axis_tdata => TEST_s_mac0_s_axis_tdata,
            TEST_s_mac0_s_axis_tlast => TEST_s_mac0_s_axis_tlast,
            TEST_s_mac0_s_axis_tvalid => TEST_s_mac0_s_axis_tvalid,
            TEST_s_mac1_s_axis_tready => TEST_s_mac1_s_axis_tready,
            TEST_s_mac1_s_axis_tdata => TEST_s_mac1_s_axis_tdata,
            TEST_s_mac1_s_axis_tlast => TEST_s_mac1_s_axis_tlast,
            TEST_s_mac1_s_axis_tvalid => TEST_s_mac1_s_axis_tvalid,
            TEST_s_mac2_s_axis_tready => TEST_s_mac2_s_axis_tready,
            TEST_s_mac2_s_axis_tdata => TEST_s_mac2_s_axis_tdata,
            TEST_s_mac2_s_axis_tlast => TEST_s_mac2_s_axis_tlast,
            TEST_s_mac2_s_axis_tvalid => TEST_s_mac2_s_axis_tvalid,
            TEST_s_mac3_s_axis_tready => TEST_s_mac3_s_axis_tready,
            TEST_s_mac3_s_axis_tdata => TEST_s_mac3_s_axis_tdata,
            TEST_s_mac3_s_axis_tlast => TEST_s_mac3_s_axis_tlast,
            TEST_s_mac3_s_axis_tvalid => TEST_s_mac3_s_axis_tvalid,

            TEST_s_mac0_m_axis_tready => TEST_s_mac0_m_axis_tready,
            TEST_s_mac0_m_axis_tdata => TEST_s_mac0_m_axis_tdata,
            TEST_s_mac0_m_axis_tlast => TEST_s_mac0_m_axis_tlast,
            TEST_s_mac0_m_axis_tvalid => TEST_s_mac0_m_axis_tvalid,
            TEST_s_mac1_m_axis_tready => TEST_s_mac1_m_axis_tready,
            TEST_s_mac1_m_axis_tdata => TEST_s_mac1_m_axis_tdata,
            TEST_s_mac1_m_axis_tlast => TEST_s_mac1_m_axis_tlast,
            TEST_s_mac1_m_axis_tvalid => TEST_s_mac1_m_axis_tvalid,
            TEST_s_mac2_m_axis_tready => TEST_s_mac2_m_axis_tready,
            TEST_s_mac2_m_axis_tdata => TEST_s_mac2_m_axis_tdata,
            TEST_s_mac2_m_axis_tlast => TEST_s_mac2_m_axis_tlast,
            TEST_s_mac2_m_axis_tvalid => TEST_s_mac2_m_axis_tvalid,
            TEST_s_mac3_m_axis_tready => TEST_s_mac3_m_axis_tready,
            TEST_s_mac3_m_axis_tdata => TEST_s_mac3_m_axis_tdata,
            TEST_s_mac3_m_axis_tlast => TEST_s_mac3_m_axis_tlast,
            TEST_s_mac3_m_axis_tvalid => TEST_s_mac3_m_axis_tvalid,

            TEST_s_out_combiner_m_axis_tready => TEST_s_out_combiner_m_axis_tready,
            TEST_s_out_combiner_m_axis_tdata => TEST_s_out_combiner_m_axis_tdata,
            TEST_s_out_combiner_m_axis_tlast => TEST_s_out_combiner_m_axis_tlast,
            TEST_s_out_combiner_m_axis_tid => TEST_s_out_combiner_m_axis_tid,
            TEST_s_out_combiner_m_axis_tvalid => TEST_s_out_combiner_m_axis_tvalid,

            TEST_s_dequantization_m_axis_tready => TEST_s_dequantization_m_axis_tready,
            TEST_s_dequantization_m_axis_tdata => TEST_s_dequantization_m_axis_tdata,
            TEST_s_dequantization_m_axis_tlast => TEST_s_dequantization_m_axis_tlast,
            TEST_s_dequantization_m_axis_tid => TEST_s_dequantization_m_axis_tid,
            TEST_s_dequantization_m_axis_tvalid => TEST_s_dequantization_m_axis_tvalid,

            conv_complete => conv_complete,
            conv_idle => conv_idle,
            rst => rst,
            clk => clk
        );

    process begin
        rst <= '1';
        wait for 2ps;
        conv_idle <= '1';
        rst <= '0';
        {indent(control_process, 2)}

        assert FALSE Report "Simulation Complete!" severity FAILURE;
    end process;

    {indent(gen_axis_checking_process(
        's_index_gen_m_axis', 
        [
            ('s_index_gen_m_axis_tdata_input_addr', index_gen_input_addr, 7), 
            ('s_index_gen_m_axis_tdata_filter_addr', index_gen_filter_addr, 7), 
            ('s_index_gen_m_axis_tlast', index_gen_tlast, 1)
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac0_s_axis', 
        [
            ('s_mac0_s_axis_tdata', mac0_in_tdata, 16), 
            ('s_mac0_s_axis_tlast', mac0_in_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac1_s_axis', 
        [
            ('s_mac1_s_axis_tdata', mac1_in_tdata, 16), 
            ('s_mac1_s_axis_tlast', mac1_in_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac2_s_axis', 
        [
            ('s_mac2_s_axis_tdata', mac2_in_tdata, 16), 
            ('s_mac2_s_axis_tlast', mac2_in_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac3_s_axis', 
        [
            ('s_mac3_s_axis_tdata', mac3_in_tdata, 16), 
            ('s_mac3_s_axis_tlast', mac3_in_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac0_m_axis', 
        [
            ('s_mac0_m_axis_tdata', mac0_out_tdata, 32), 
            ('s_mac0_m_axis_tlast', mac0_out_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac1_m_axis', 
        [
            ('s_mac1_m_axis_tdata', mac1_out_tdata, 32), 
            ('s_mac1_m_axis_tlast', mac1_out_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac2_m_axis', 
        [
            ('s_mac2_m_axis_tdata', mac2_out_tdata, 32), 
            ('s_mac2_m_axis_tlast', mac2_out_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_mac3_m_axis', 
        [
            ('s_mac3_m_axis_tdata', mac3_out_tdata, 32), 
            ('s_mac3_m_axis_tlast', mac3_out_tlast, 1), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_out_combiner_m_axis', 
        [
            ('s_out_combiner_m_axis_tdata', combined_out_tdata, 32), 
            ('s_out_combiner_m_axis_tlast', combined_out_tlast, 1), 
            ('s_out_combiner_m_axis_tid', combined_out_tid, 2), 
        ]), 1)}

    {indent(gen_axis_checking_process(
        's_dequantization_m_axis',
        [
            ('s_dequantization_m_axis_tdata', deq_out_tdata, 8),
            ('s_dequantization_m_axis_tlast', deq_out_tlast, 1),
            ('s_dequantization_m_axis_tid', deq_out_tid, 2),
        ]), 1)}

    {indent(gen_bram_checking_process(
        'BRAM_OUTPUT',
        [
            ('BRAM_OUTPUT_addr', bram_output_write_addr, 32),
            ('BRAM_OUTPUT_din', bram_output_write_data, 32),
        ]), 1)}

end Behavioral;
""")
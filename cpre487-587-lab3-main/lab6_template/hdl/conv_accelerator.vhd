----------------------------------------------------------------------------------
-- Full Accelerator without config unit (for use with testbench)
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity conv_accelerator is
    generic(
        DIM_WIDTH : integer := 12; -- Max dim size is 2048 in a dense layer
        INPUT_ADDR_WIDTH : integer := 17; -- Max input size is 60*60*32 < 2^17
        FILTER_ADDR_WIDTH : integer := 11; -- Max filter size is 2048 = 2^11
        OUTPUT_ADDR_WIDTH : integer := 17; -- Max output size is 60*60*32 < 2^17
        INPUT_BRAM_ADDR_WIDTH : integer := 15; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
        FILTER_BRAM_ADDR_WIDTH : integer := 9; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
        OUTPUT_BRAM_ADDR_WIDTH : integer := 12; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
        BRAM_DATA_WIDTH : integer := 32; -- Data width of raw BRAM interface
        MAC_DATA_WIDTH : integer := 8; -- Data width of each MAC input operand, defaults to int8. Supports sub-byte indexing, must be power of 2 and less than BRAM_DATA_WIDTH
        MAC_OUTPUT_DATA_WIDTH : integer := 32 -- Data width of the raw output of the MAC unit
    );
    port(
        -- Configuration values from conv_config unit
        max_pooling : in std_logic;
        relu : in std_logic;
        filter_w : in std_logic_vector(DIM_WIDTH-1 downto 0);
        filter_h : in std_logic_vector(DIM_WIDTH-1 downto 0);
        filter_c : in std_logic_vector(DIM_WIDTH-1 downto 0);
        output_w : in std_logic_vector(DIM_WIDTH-1 downto 0);
        output_h : in std_logic_vector(DIM_WIDTH-1 downto 0);
        input_end_diff_fw : in std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        input_end_diff_fh : in std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        input_end_diff_fc : in std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        input_end_diff_ow : in std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        output_elements_per_channel : in std_logic_vector(OUTPUT_ADDR_WIDTH-1 downto 0);
        output_initial_offset : in std_logic_vector(OUTPUT_ADDR_WIDTH-1 downto 0);
        mac0_bias : in std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        mac2_bias : in std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        mac1_bias : in std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        mac3_bias : in std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        q_scale : in std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        q_zero : in std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
        
        -- BRAM blocks for high speed memory access
        BRAM_INPUT_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_INPUT_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_INPUT_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_INPUT_en : out std_logic;
        BRAM_INPUT_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_INPUT_rst : out std_logic;
        BRAM_INPUT_clk : out std_logic;

        BRAM_FILTER0_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_FILTER0_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER0_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER0_en : out std_logic;
        BRAM_FILTER0_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_FILTER0_rst : out std_logic;
        BRAM_FILTER0_clk : out std_logic;

        BRAM_FILTER1_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_FILTER1_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER1_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER1_en : out std_logic;
        BRAM_FILTER1_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_FILTER1_rst : out std_logic;
        BRAM_FILTER1_clk : out std_logic;

        BRAM_FILTER2_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_FILTER2_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER2_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER2_en : out std_logic;
        BRAM_FILTER2_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_FILTER2_rst : out std_logic;
        BRAM_FILTER2_clk : out std_logic;

        BRAM_FILTER3_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_FILTER3_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER3_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_FILTER3_en : out std_logic;
        BRAM_FILTER3_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_FILTER3_rst : out std_logic;
        BRAM_FILTER3_clk : out std_logic;

        BRAM_OUTPUT_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_OUTPUT_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_OUTPUT_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_OUTPUT_en : out std_logic;
        BRAM_OUTPUT_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_OUTPUT_rst : out std_logic;
        BRAM_OUTPUT_clk : out std_logic;

        TEST_s_index_gen_m_axis_tready : out std_logic;
        TEST_s_index_gen_m_axis_tdata_input_addr : out std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        TEST_s_index_gen_m_axis_tdata_filter_addr : out std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
        TEST_s_index_gen_m_axis_tlast : out std_logic;
        TEST_s_index_gen_m_axis_tvalid : out std_logic;    

        TEST_s_mac0_s_axis_tready : out std_logic;
        TEST_s_mac0_s_axis_tdata : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        TEST_s_mac0_s_axis_tlast : out std_logic;
        TEST_s_mac0_s_axis_tvalid : out std_logic;
        TEST_s_mac1_s_axis_tready : out std_logic;
        TEST_s_mac1_s_axis_tdata : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        TEST_s_mac1_s_axis_tlast : out std_logic;
        TEST_s_mac1_s_axis_tvalid : out std_logic;
        TEST_s_mac2_s_axis_tready : out std_logic;
        TEST_s_mac2_s_axis_tdata : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        TEST_s_mac2_s_axis_tlast : out std_logic;
        TEST_s_mac2_s_axis_tvalid : out std_logic;
        TEST_s_mac3_s_axis_tready : out std_logic;
        TEST_s_mac3_s_axis_tdata : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        TEST_s_mac3_s_axis_tlast : out std_logic;
        TEST_s_mac3_s_axis_tvalid : out std_logic;

        TEST_s_mac0_m_axis_tready : out std_logic;
        TEST_s_mac0_m_axis_tdata : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        TEST_s_mac0_m_axis_tlast : out std_logic;
        TEST_s_mac0_m_axis_tvalid : out std_logic;
        TEST_s_mac1_m_axis_tready : out std_logic;
        TEST_s_mac1_m_axis_tdata : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        TEST_s_mac1_m_axis_tlast : out std_logic;
        TEST_s_mac1_m_axis_tvalid : out std_logic;
        TEST_s_mac2_m_axis_tready : out std_logic;
        TEST_s_mac2_m_axis_tdata : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        TEST_s_mac2_m_axis_tlast : out std_logic;
        TEST_s_mac2_m_axis_tvalid : out std_logic;
        TEST_s_mac3_m_axis_tready : out std_logic;
        TEST_s_mac3_m_axis_tdata : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        TEST_s_mac3_m_axis_tlast : out std_logic;
        TEST_s_mac3_m_axis_tvalid : out std_logic;

        TEST_s_out_combiner_m_axis_tready : out std_logic;
        TEST_s_out_combiner_m_axis_tdata : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        TEST_s_out_combiner_m_axis_tlast : out std_logic;
        TEST_s_out_combiner_m_axis_tid : out std_logic_vector(1 downto 0);
        TEST_s_out_combiner_m_axis_tvalid : out std_logic;

        TEST_s_dequantization_m_axis_tready : out std_logic;
        TEST_s_dequantization_m_axis_tdata : out std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
        TEST_s_dequantization_m_axis_tlast : out std_logic;
        TEST_s_dequantization_m_axis_tid : out std_logic_vector(1 downto 0);
        TEST_s_dequantization_m_axis_tvalid : out std_logic;

        conv_complete : out std_logic;
        conv_idle : in std_logic;
        rst : in std_logic; -- Reset everything, including BRAM contents
        clk : in std_logic
    );
end conv_accelerator;

architecture Behavioral of conv_accelerator is

    signal s_bram_en : std_logic;
    
    signal s_index_gen_m_axis_tready : std_logic;
    signal s_index_gen_m_axis_tdata_input_addr : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_index_gen_m_axis_tdata_filter_addr : std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
    signal s_index_gen_m_axis_tlast : std_logic;
    signal s_index_gen_m_axis_tvalid : std_logic;    

    signal s_mac0_s_axis_tready : std_logic;
    signal s_mac0_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal s_mac0_s_axis_tlast : std_logic;
    signal s_mac0_s_axis_tvalid : std_logic;
    signal s_mac1_s_axis_tready : std_logic;
    signal s_mac1_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal s_mac1_s_axis_tlast : std_logic;
    signal s_mac1_s_axis_tvalid : std_logic;
    signal s_mac2_s_axis_tready : std_logic;
    signal s_mac2_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal s_mac2_s_axis_tlast : std_logic;
    signal s_mac2_s_axis_tvalid : std_logic;
    signal s_mac3_s_axis_tready : std_logic;
    signal s_mac3_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
    signal s_mac3_s_axis_tlast : std_logic;
    signal s_mac3_s_axis_tvalid : std_logic;

    signal s_mac0_m_axis_tready : std_logic;
    signal s_mac0_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac0_m_axis_tlast : std_logic;
    signal s_mac0_m_axis_tvalid : std_logic;
    signal s_mac1_m_axis_tready : std_logic;
    signal s_mac1_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac1_m_axis_tlast : std_logic;
    signal s_mac1_m_axis_tvalid : std_logic;
    signal s_mac2_m_axis_tready : std_logic;
    signal s_mac2_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac2_m_axis_tlast : std_logic;
    signal s_mac2_m_axis_tvalid : std_logic;
    signal s_mac3_m_axis_tready : std_logic;
    signal s_mac3_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac3_m_axis_tlast : std_logic;
    signal s_mac3_m_axis_tvalid : std_logic;

    signal s_out_combiner_m_axis_tready : std_logic;
    signal s_out_combiner_m_axis_tdata : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_out_combiner_m_axis_tlast : std_logic;
    signal s_out_combiner_m_axis_tid : std_logic_vector(1 downto 0);
    signal s_out_combiner_m_axis_tvalid : std_logic;

    signal s_dequantization_m_axis_tready : std_logic;
    signal s_dequantization_m_axis_tdata : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_dequantization_m_axis_tlast : std_logic;
    signal s_dequantization_m_axis_tid : std_logic_vector(1 downto 0);
    signal s_dequantization_m_axis_tvalid : std_logic;

begin

    TEST_s_index_gen_m_axis_tready <= s_index_gen_m_axis_tready;
    TEST_s_index_gen_m_axis_tdata_input_addr <= s_index_gen_m_axis_tdata_input_addr;
    TEST_s_index_gen_m_axis_tdata_filter_addr <= s_index_gen_m_axis_tdata_filter_addr;
    TEST_s_index_gen_m_axis_tlast <= s_index_gen_m_axis_tlast;
    TEST_s_index_gen_m_axis_tvalid <= s_index_gen_m_axis_tvalid;

    TEST_s_mac0_s_axis_tready <= s_mac0_s_axis_tready;
    TEST_s_mac0_s_axis_tdata <= s_mac0_s_axis_tdata;
    TEST_s_mac0_s_axis_tlast <= s_mac0_s_axis_tlast;
    TEST_s_mac0_s_axis_tvalid <= s_mac0_s_axis_tvalid;
    TEST_s_mac1_s_axis_tready <= s_mac1_s_axis_tready;
    TEST_s_mac1_s_axis_tdata <= s_mac1_s_axis_tdata;
    TEST_s_mac1_s_axis_tlast <= s_mac1_s_axis_tlast;
    TEST_s_mac1_s_axis_tvalid <= s_mac1_s_axis_tvalid;
    TEST_s_mac2_s_axis_tready <= s_mac2_s_axis_tready;
    TEST_s_mac2_s_axis_tdata <= s_mac2_s_axis_tdata;
    TEST_s_mac2_s_axis_tlast <= s_mac2_s_axis_tlast;
    TEST_s_mac2_s_axis_tvalid <= s_mac2_s_axis_tvalid;
    TEST_s_mac3_s_axis_tready <= s_mac3_s_axis_tready;
    TEST_s_mac3_s_axis_tdata <= s_mac3_s_axis_tdata;
    TEST_s_mac3_s_axis_tlast <= s_mac3_s_axis_tlast;
    TEST_s_mac3_s_axis_tvalid <= s_mac3_s_axis_tvalid;
    TEST_s_mac0_m_axis_tready <= s_mac0_m_axis_tready;
    TEST_s_mac0_m_axis_tdata <= s_mac0_m_axis_tdata;
    TEST_s_mac0_m_axis_tlast <= s_mac0_m_axis_tlast;
    TEST_s_mac0_m_axis_tvalid <= s_mac0_m_axis_tvalid;
    TEST_s_mac1_m_axis_tready <= s_mac1_m_axis_tready;
    TEST_s_mac1_m_axis_tdata <= s_mac1_m_axis_tdata;
    TEST_s_mac1_m_axis_tlast <= s_mac1_m_axis_tlast;
    TEST_s_mac1_m_axis_tvalid <= s_mac1_m_axis_tvalid;
    TEST_s_mac2_m_axis_tready <= s_mac2_m_axis_tready;
    TEST_s_mac2_m_axis_tdata <= s_mac2_m_axis_tdata;
    TEST_s_mac2_m_axis_tlast <= s_mac2_m_axis_tlast;
    TEST_s_mac2_m_axis_tvalid <= s_mac2_m_axis_tvalid;
    TEST_s_mac3_m_axis_tready <= s_mac3_m_axis_tready;
    TEST_s_mac3_m_axis_tdata <= s_mac3_m_axis_tdata;
    TEST_s_mac3_m_axis_tlast <= s_mac3_m_axis_tlast;
    TEST_s_mac3_m_axis_tvalid <= s_mac3_m_axis_tvalid;

    TEST_s_out_combiner_m_axis_tready <= s_out_combiner_m_axis_tready;
    TEST_s_out_combiner_m_axis_tdata <= s_out_combiner_m_axis_tdata;
    TEST_s_out_combiner_m_axis_tlast <= s_out_combiner_m_axis_tlast;
    TEST_s_out_combiner_m_axis_tid <= s_out_combiner_m_axis_tid;
    TEST_s_out_combiner_m_axis_tvalid <= s_out_combiner_m_axis_tvalid;

    TEST_s_dequantization_m_axis_tready <= s_dequantization_m_axis_tready;
    TEST_s_dequantization_m_axis_tdata <= s_dequantization_m_axis_tdata;
    TEST_s_dequantization_m_axis_tlast <= s_dequantization_m_axis_tlast;
    TEST_s_dequantization_m_axis_tid <= s_dequantization_m_axis_tid;
    TEST_s_dequantization_m_axis_tvalid <= s_dequantization_m_axis_tvalid;

    -- BRAM Index Generation

    g_index_gen: entity work.index_gen
        generic map(
            DIM_WIDTH => DIM_WIDTH,
            INPUT_ADDR_WIDTH => INPUT_ADDR_WIDTH,
            FILTER_ADDR_WIDTH => FILTER_ADDR_WIDTH,
            OUTPUT_ADDR_WIDTH => OUTPUT_ADDR_WIDTH
        )
        port map(
            filter_w => filter_w,
            filter_h => filter_h,
            filter_c => filter_c,
            output_w => output_w,
            output_h => output_h,
            input_end_diff_fw => input_end_diff_fw,
            input_end_diff_fh => input_end_diff_fh,
            input_end_diff_fc => input_end_diff_fc,
            input_end_diff_ow => input_end_diff_ow,
            
            M_AXIS_TREADY => s_index_gen_m_axis_tready,
            M_AXIS_TDATA_input_addr => s_index_gen_m_axis_tdata_input_addr,
            M_AXIS_TDATA_filter_addr => s_index_gen_m_axis_tdata_filter_addr,
            M_AXIS_TLAST => s_index_gen_m_axis_tlast,
            M_AXIS_TVALID => s_index_gen_m_axis_tvalid,
            
            conv_idle => conv_idle,
            rst => rst,
            clk => clk
        );

    -- Master AXI Streams to MAC Units

    g_mac_stream_provider: entity work.mac_stream_provider
        generic map(
            INPUT_ADDR_WIDTH => INPUT_ADDR_WIDTH,
            FILTER_ADDR_WIDTH => FILTER_ADDR_WIDTH,
            INPUT_BRAM_ADDR_WIDTH => INPUT_BRAM_ADDR_WIDTH,
            FILTER_BRAM_ADDR_WIDTH => FILTER_BRAM_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            MAC_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            S_AXIS_TREADY => s_index_gen_m_axis_tready,
            S_AXIS_TDATA_input_addr => s_index_gen_m_axis_tdata_input_addr,
            S_AXIS_TDATA_filter_addr => s_index_gen_m_axis_tdata_filter_addr,
            S_AXIS_TLAST => s_index_gen_m_axis_tlast,
            S_AXIS_TVALID => s_index_gen_m_axis_tvalid,

            BRAM_INPUT_addr => BRAM_INPUT_addr,
            BRAM_INPUT_din => BRAM_INPUT_din,
            BRAM_INPUT_dout => BRAM_INPUT_dout,
            BRAM_INPUT_en => BRAM_INPUT_en,
            BRAM_INPUT_we => BRAM_INPUT_we,
            BRAM_INPUT_rst => BRAM_INPUT_rst,
            BRAM_INPUT_clk => BRAM_INPUT_clk,

            BRAM_FILTER0_addr => BRAM_FILTER0_addr,
            BRAM_FILTER0_din => BRAM_FILTER0_din,
            BRAM_FILTER0_dout => BRAM_FILTER0_dout,
            BRAM_FILTER0_en => BRAM_FILTER0_en,
            BRAM_FILTER0_we => BRAM_FILTER0_we,
            BRAM_FILTER0_rst => BRAM_FILTER0_rst,
            BRAM_FILTER0_clk => BRAM_FILTER0_clk,

            BRAM_FILTER1_addr => BRAM_FILTER1_addr,
            BRAM_FILTER1_din => BRAM_FILTER1_din,
            BRAM_FILTER1_dout => BRAM_FILTER1_dout,
            BRAM_FILTER1_en => BRAM_FILTER1_en,
            BRAM_FILTER1_we => BRAM_FILTER1_we,
            BRAM_FILTER1_rst => BRAM_FILTER1_rst,
            BRAM_FILTER1_clk => BRAM_FILTER1_clk,

            BRAM_FILTER2_addr => BRAM_FILTER2_addr,
            BRAM_FILTER2_din => BRAM_FILTER2_din,
            BRAM_FILTER2_dout => BRAM_FILTER2_dout,
            BRAM_FILTER2_en => BRAM_FILTER2_en,
            BRAM_FILTER2_we => BRAM_FILTER2_we,
            BRAM_FILTER2_rst => BRAM_FILTER2_rst,
            BRAM_FILTER2_clk => BRAM_FILTER2_clk,

            BRAM_FILTER3_addr => BRAM_FILTER3_addr,
            BRAM_FILTER3_din => BRAM_FILTER3_din,
            BRAM_FILTER3_dout => BRAM_FILTER3_dout,
            BRAM_FILTER3_en => BRAM_FILTER3_en,
            BRAM_FILTER3_we => BRAM_FILTER3_we,
            BRAM_FILTER3_rst => BRAM_FILTER3_rst,
            BRAM_FILTER3_clk => BRAM_FILTER3_clk,

            M_AXIS_MAC0_TREADY => s_mac0_s_axis_tready,
            M_AXIS_MAC0_TDATA => s_mac0_s_axis_tdata,
            M_AXIS_MAC0_TLAST => s_mac0_s_axis_tlast,
            M_AXIS_MAC0_TVALID => s_mac0_s_axis_tvalid,

            M_AXIS_MAC1_TREADY => s_mac1_s_axis_tready,
            M_AXIS_MAC1_TDATA => s_mac1_s_axis_tdata,
            M_AXIS_MAC1_TLAST => s_mac1_s_axis_tlast,
            M_AXIS_MAC1_TVALID => s_mac1_s_axis_tvalid,

            M_AXIS_MAC2_TREADY => s_mac2_s_axis_tready,
            M_AXIS_MAC2_TDATA => s_mac2_s_axis_tdata,
            M_AXIS_MAC2_TLAST => s_mac2_s_axis_tlast,
            M_AXIS_MAC2_TVALID => s_mac2_s_axis_tvalid,

            M_AXIS_MAC3_TREADY => s_mac3_s_axis_tready,
            M_AXIS_MAC3_TDATA => s_mac3_s_axis_tdata,
            M_AXIS_MAC3_TLAST => s_mac3_s_axis_tlast,
            M_AXIS_MAC3_TVALID => s_mac3_s_axis_tvalid,

            rst => rst,
            clk => clk
        );
    
    g_mac0: entity work.conv_mac
        generic map(
            C_DATA_WIDTH => MAC_DATA_WIDTH,
            C_OUTPUT_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH
        )
        port map(
            -- AXIS slave data interface
            S_AXIS_TREADY => s_mac0_s_axis_tready,
            S_AXIS_TDATA => s_mac0_s_axis_tdata,
            S_AXIS_TLAST => s_mac0_s_axis_tlast,
            S_AXIS_TVALID => s_mac0_s_axis_tvalid,

            bias => mac0_bias,
    
            -- AXIS master accumulate result out interface
            M_AXIS_TVALID => s_mac0_m_axis_tvalid,
            M_AXIS_TDATA => s_mac0_m_axis_tdata,
            M_AXIS_TLAST => s_mac0_m_axis_tlast,
            M_AXIS_TREADY => s_mac0_m_axis_tready,

            rst => rst,
            clk => clk
        );

    g_mac1: entity work.conv_mac
        generic map(
            C_DATA_WIDTH => MAC_DATA_WIDTH,
            C_OUTPUT_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH
        )
        port map(
            -- AXIS slave data interface
            S_AXIS_TREADY => s_mac1_s_axis_tready,
            S_AXIS_TDATA => s_mac1_s_axis_tdata,
            S_AXIS_TLAST => s_mac1_s_axis_tlast,
            S_AXIS_TVALID => s_mac1_s_axis_tvalid,

            bias => mac1_bias,
    
            -- AXIS master accumulate result out interface
            M_AXIS_TREADY => s_mac1_m_axis_tready,
            M_AXIS_TDATA => s_mac1_m_axis_tdata,
            M_AXIS_TLAST => s_mac1_m_axis_tlast,
            M_AXIS_TVALID => s_mac1_m_axis_tvalid,

            rst => rst,
            clk => clk
        );

    g_mac2: entity work.conv_mac
        generic map(
            C_DATA_WIDTH => MAC_DATA_WIDTH,
            C_OUTPUT_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH
        )
        port map(
            -- AXIS slave data interface
            S_AXIS_TREADY => s_mac2_s_axis_tready,
            S_AXIS_TDATA => s_mac2_s_axis_tdata,
            S_AXIS_TLAST => s_mac2_s_axis_tlast,
            S_AXIS_TVALID => s_mac2_s_axis_tvalid,

            bias => mac2_bias,
    
            -- AXIS master accumulate result out interface
            M_AXIS_TREADY => s_mac2_m_axis_tready,
            M_AXIS_TDATA => s_mac2_m_axis_tdata,
            M_AXIS_TLAST => s_mac2_m_axis_tlast,
            M_AXIS_TVALID => s_mac2_m_axis_tvalid,

            rst => rst,
            clk => clk
        );

    g_mac3: entity work.conv_mac
        generic map(
            C_DATA_WIDTH => MAC_DATA_WIDTH,
            C_OUTPUT_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH
        )
        port map(
            -- AXIS slave data interface
            S_AXIS_TREADY => s_mac3_s_axis_tready,
            S_AXIS_TDATA => s_mac3_s_axis_tdata,
            S_AXIS_TLAST => s_mac3_s_axis_tlast,
            S_AXIS_TVALID => s_mac3_s_axis_tvalid,

            bias => mac3_bias,
    
            -- AXIS master accumulate result out interface
            M_AXIS_TREADY => s_mac3_m_axis_tready,
            M_AXIS_TDATA => s_mac3_m_axis_tdata,
            M_AXIS_TLAST => s_mac3_m_axis_tlast,
            M_AXIS_TVALID => s_mac3_m_axis_tvalid,

            rst => rst,
            clk => clk
        );

    -- Output to BRAM and mark convolution complete when done

    -- Combine the AXI streams from the MAC units into one stream to pass through the dequantization and output storage logic
    -- Note that this buffers the data in a register first so the MAC units are not held up by the dequantization stage.
    g_out_combiner: entity work.axis_buffered_4_to_1_round_robin_combiner
        generic map(
            C_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH
        )
        port map(
            S_AXIS0_TREADY => s_mac0_m_axis_tready,
            S_AXIS0_TDATA => s_mac0_m_axis_tdata,
            S_AXIS0_TLAST => s_mac0_m_axis_tlast,
            S_AXIS0_TVALID => s_mac0_m_axis_tvalid,

            S_AXIS1_TREADY => s_mac1_m_axis_tready,
            S_AXIS1_TDATA => s_mac1_m_axis_tdata,
            S_AXIS1_TLAST => s_mac1_m_axis_tlast,
            S_AXIS1_TVALID => s_mac1_m_axis_tvalid,

            S_AXIS2_TREADY => s_mac2_m_axis_tready,
            S_AXIS2_TDATA => s_mac2_m_axis_tdata,
            S_AXIS2_TLAST => s_mac2_m_axis_tlast,
            S_AXIS2_TVALID => s_mac2_m_axis_tvalid,

            S_AXIS3_TREADY => s_mac3_m_axis_tready,
            S_AXIS3_TDATA => s_mac3_m_axis_tdata,
            S_AXIS3_TLAST => s_mac3_m_axis_tlast,
            S_AXIS3_TVALID => s_mac3_m_axis_tvalid,

            M_AXIS_TREADY => s_out_combiner_m_axis_tready,
            M_AXIS_TDATA => s_out_combiner_m_axis_tdata,
            M_AXIS_TLAST => s_out_combiner_m_axis_tlast,
            M_AXIS_TID => s_out_combiner_m_axis_tid,
            M_AXIS_TVALID => s_out_combiner_m_axis_tvalid,

            clk => clk,
            rst => rst
        );

   g_dequantization: entity work.dequantization
        generic map(
            C_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH,
            C_TID_WIDTH => 2,
            C_OUT_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            S_AXIS_TREADY => s_out_combiner_m_axis_tready,
            S_AXIS_TDATA => s_out_combiner_m_axis_tdata,
            S_AXIS_TLAST => s_out_combiner_m_axis_tlast,
            S_AXIS_TID => s_out_combiner_m_axis_tid,
            S_AXIS_TVALID => s_out_combiner_m_axis_tvalid,

            relu => relu,
            q_scale => q_scale,
            q_zero => q_zero,

            M_AXIS_TREADY => s_dequantization_m_axis_tready,
            M_AXIS_TDATA => s_dequantization_m_axis_tdata,
            M_AXIS_TLAST => s_dequantization_m_axis_tlast,
            M_AXIS_TID => s_dequantization_m_axis_tid,
            M_AXIS_TVALID => s_dequantization_m_axis_tvalid,

            clk => clk,
            rst => rst
        );

    g_output_storage: entity work.output_storage
        generic map(
            DATA_WIDTH => MAC_DATA_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            ADDR_WIDTH => OUTPUT_ADDR_WIDTH,
            BRAM_ADDR_WIDTH => OUTPUT_BRAM_ADDR_WIDTH,
            DIM_WIDTH => DIM_WIDTH,
            C_TID_WIDTH => 2
        )
        port map(
            S_AXIS_TREADY => s_dequantization_m_axis_tready,
            S_AXIS_TDATA => s_dequantization_m_axis_tdata,
            S_AXIS_TLAST => s_dequantization_m_axis_tlast,
            S_AXIS_TID => s_dequantization_m_axis_tid,
            S_AXIS_TVALID => s_dequantization_m_axis_tvalid,

            BRAM_addr => BRAM_OUTPUT_addr,
            BRAM_din => BRAM_OUTPUT_din,
            BRAM_dout => BRAM_OUTPUT_dout,
            BRAM_en => BRAM_OUTPUT_en,
            BRAM_we => BRAM_OUTPUT_we,
            BRAM_rst => BRAM_OUTPUT_rst,
            BRAM_clk => BRAM_OUTPUT_clk,

            max_pooling => max_pooling,
            elements_per_channel => output_elements_per_channel,
            output_w => output_w,
            output_h => output_h,
            initial_offset => output_initial_offset,
            
            conv_complete => conv_complete,
            conv_idle => conv_idle,
            clk => clk,
            rst => rst
        );

end Behavioral;

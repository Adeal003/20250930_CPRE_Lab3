----------------------------------------------------------------------------------
-- Full Accelerator Wrapper with config unit
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity conv_accelerator_wrapper is
    generic(
        C_AXI_ADDR_WIDTH : integer := 8; -- 2^(C_AXI_ADDR_WIDTH-2) registers
        C_AXI_DATA_WIDTH : integer := 32; -- Fixed by AXI-lite spec
        DIM_WIDTH : integer := 12; -- Max dim size is 2048 in a dense layer
        INPUT_ADDR_WIDTH : integer := 17; -- Max input size is 60*60*32 < 2^17
        FILTER_ADDR_WIDTH : integer := 11; -- Max filter size is 2048 <= 2^11
        OUTPUT_ADDR_WIDTH : integer := 17; -- Max output size is 60*60*32 < 2^17
        INPUT_BRAM_ADDR_WIDTH : integer := 15; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
        FILTER_BRAM_ADDR_WIDTH : integer := 9; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
        OUTPUT_BRAM_ADDR_WIDTH : integer := 15; -- Word address width to the BRAM interfaces, must be kept in sync with ADDR_WIDTH, BRAM_DATA_WIDTH, and MAC_DATA_WIDTH!!!
        BRAM_DATA_WIDTH : integer := 32; -- Data width of raw BRAM interface
        MAC_DATA_WIDTH : integer := 8; -- Data width of each MAC input operand, defaults to int8. Supports sub-byte indexing, must be power of 2 and less than BRAM_DATA_WIDTH
        MAC_OUTPUT_DATA_WIDTH : integer := 32 -- Data width of the raw output of the MAC unit
    );
    port(
        -- AXI4LITE interface for configuration
        S_AXI_LITE_ACLK : in std_logic;
        S_AXI_LITE_ARESETN : in std_logic;
        S_AXI_LITE_AWVALID : in std_logic;
        S_AXI_LITE_AWREADY : out std_logic;
        S_AXI_LITE_AWADDR : in std_logic_vector(C_AXI_ADDR_WIDTH-1 downto 0);
        S_AXI_LITE_AWPROT : in std_logic_vector(2 downto 0);
        S_AXI_LITE_WVALID : in std_logic;
        S_AXI_LITE_WREADY : out std_logic;
        S_AXI_LITE_WDATA : in std_logic_vector(C_AXI_DATA_WIDTH-1 downto 0);
        S_AXI_LITE_WSTRB : in std_logic_vector((C_AXI_DATA_WIDTH/8)-1 downto 0);
        S_AXI_LITE_BVALID : out std_logic;
        S_AXI_LITE_BREADY : in std_logic;
        S_AXI_LITE_BRESP : out std_logic_vector(1 downto 0);
        S_AXI_LITE_ARVALID : in std_logic;
        S_AXI_LITE_ARREADY : out std_logic;
        S_AXI_LITE_ARADDR : in std_logic_vector(C_AXI_ADDR_WIDTH-1 downto 0);
        S_AXI_LITE_ARPROT : in std_logic_vector(2 downto 0);
        S_AXI_LITE_RVALID : out std_logic;
        S_AXI_LITE_RREADY : in std_logic;
        S_AXI_LITE_RDATA : out std_logic_vector(C_AXI_DATA_WIDTH-1 downto 0);
        S_AXI_LITE_RRESP : out std_logic_vector(1 downto 0);

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

        accelerator_controls_activation_bram : out std_logic;
        swap_activations : out std_logic;
        swap_filters : out std_logic
    );    
end conv_accelerator_wrapper;

architecture Behavioral of conv_accelerator_wrapper is

    attribute X_INTERFACE_INFO : STRING;
    attribute X_INTERFACE_MODE : STRING;

    attribute X_INTERFACE_INFO of BRAM_INPUT_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT ADDR";
    attribute X_INTERFACE_INFO of BRAM_INPUT_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT CLK";
    attribute X_INTERFACE_INFO of BRAM_INPUT_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT DIN";
    attribute X_INTERFACE_INFO of BRAM_INPUT_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT DOUT";
    attribute X_INTERFACE_INFO of BRAM_INPUT_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT EN";
    attribute X_INTERFACE_INFO of BRAM_INPUT_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT WE";
    attribute X_INTERFACE_INFO of BRAM_INPUT_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_INPUT RST";
    attribute X_INTERFACE_MODE of BRAM_INPUT_addr:  SIGNAL is "Master"; 

    attribute X_INTERFACE_INFO of BRAM_FILTER0_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 ADDR";
    attribute X_INTERFACE_INFO of BRAM_FILTER0_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 CLK";
    attribute X_INTERFACE_INFO of BRAM_FILTER0_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 DIN";
    attribute X_INTERFACE_INFO of BRAM_FILTER0_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 DOUT";
    attribute X_INTERFACE_INFO of BRAM_FILTER0_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 EN";
    attribute X_INTERFACE_INFO of BRAM_FILTER0_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 WE";
    attribute X_INTERFACE_INFO of BRAM_FILTER0_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER0 RST";
    attribute X_INTERFACE_MODE of BRAM_FILTER0_addr:  SIGNAL is "Master"; 

    attribute X_INTERFACE_INFO of BRAM_FILTER1_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 ADDR";
    attribute X_INTERFACE_INFO of BRAM_FILTER1_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 CLK";
    attribute X_INTERFACE_INFO of BRAM_FILTER1_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 DIN";
    attribute X_INTERFACE_INFO of BRAM_FILTER1_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 DOUT";
    attribute X_INTERFACE_INFO of BRAM_FILTER1_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 EN";
    attribute X_INTERFACE_INFO of BRAM_FILTER1_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 WE";
    attribute X_INTERFACE_INFO of BRAM_FILTER1_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER1 RST";
    attribute X_INTERFACE_MODE of BRAM_FILTER1_addr:  SIGNAL is "Master"; 

    attribute X_INTERFACE_INFO of BRAM_FILTER2_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 ADDR";
    attribute X_INTERFACE_INFO of BRAM_FILTER2_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 CLK";
    attribute X_INTERFACE_INFO of BRAM_FILTER2_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 DIN";
    attribute X_INTERFACE_INFO of BRAM_FILTER2_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 DOUT";
    attribute X_INTERFACE_INFO of BRAM_FILTER2_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 EN";
    attribute X_INTERFACE_INFO of BRAM_FILTER2_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 WE";
    attribute X_INTERFACE_INFO of BRAM_FILTER2_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER2 RST";
    attribute X_INTERFACE_MODE of BRAM_FILTER2_addr:  SIGNAL is "Master"; 

    attribute X_INTERFACE_INFO of BRAM_FILTER3_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 ADDR";
    attribute X_INTERFACE_INFO of BRAM_FILTER3_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 CLK";
    attribute X_INTERFACE_INFO of BRAM_FILTER3_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 DIN";
    attribute X_INTERFACE_INFO of BRAM_FILTER3_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 DOUT";
    attribute X_INTERFACE_INFO of BRAM_FILTER3_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 EN";
    attribute X_INTERFACE_INFO of BRAM_FILTER3_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 WE";
    attribute X_INTERFACE_INFO of BRAM_FILTER3_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_FILTER3 RST";
    attribute X_INTERFACE_MODE of BRAM_FILTER3_addr:  SIGNAL is "Master"; 

    attribute X_INTERFACE_INFO of BRAM_OUTPUT_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT ADDR";
    attribute X_INTERFACE_INFO of BRAM_OUTPUT_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT CLK";
    attribute X_INTERFACE_INFO of BRAM_OUTPUT_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT DIN";
    attribute X_INTERFACE_INFO of BRAM_OUTPUT_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT DOUT";
    attribute X_INTERFACE_INFO of BRAM_OUTPUT_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT EN";
    attribute X_INTERFACE_INFO of BRAM_OUTPUT_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT WE";
    attribute X_INTERFACE_INFO of BRAM_OUTPUT_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_OUTPUT RST";
    attribute X_INTERFACE_MODE of BRAM_OUTPUT_addr:  SIGNAL is "Master"; 

    signal s_conv_complete : std_logic;
    signal s_conv_idle : std_logic;
    signal s_reset: std_logic;
    signal s_clk : std_logic;

    signal s_max_pooling : std_logic;
    signal s_relu : std_logic;
    signal s_filter_w : std_logic_vector(DIM_WIDTH-1 downto 0);
    signal s_filter_h : std_logic_vector(DIM_WIDTH-1 downto 0);
    signal s_filter_c : std_logic_vector(DIM_WIDTH-1 downto 0);
    signal s_output_w : std_logic_vector(DIM_WIDTH-1 downto 0);
    signal s_output_h : std_logic_vector(DIM_WIDTH-1 downto 0);
    signal s_input_end_diff_fw : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_input_end_diff_fh : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_input_end_diff_fc : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_input_end_diff_ow : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_output_elements_per_channel : std_logic_vector(OUTPUT_ADDR_WIDTH-1 downto 0);
    signal s_output_initial_offset : std_logic_vector(OUTPUT_ADDR_WIDTH-1 downto 0);
    signal s_mac0_bias : std_logic_vector(31 downto 0);
    signal s_mac1_bias : std_logic_vector(31 downto 0);
    signal s_mac2_bias : std_logic_vector(31 downto 0);
    signal s_mac3_bias : std_logic_vector(31 downto 0);
    signal s_q_scale : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_q_zero : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
begin

    s_reset <= not S_AXI_LITE_ARESETN;
    s_clk <= S_AXI_LITE_ACLK;

    -- AXI LITE configuration registers

    g_config: entity work.conv_config
        generic map(
            C_AXI_ADDR_WIDTH => C_AXI_ADDR_WIDTH,
            C_AXI_DATA_WIDTH => C_AXI_DATA_WIDTH,
            DIM_WIDTH => DIM_WIDTH,
            INPUT_ADDR_WIDTH => INPUT_ADDR_WIDTH,
            FILTER_ADDR_WIDTH => FILTER_ADDR_WIDTH,
            OUTPUT_ADDR_WIDTH => OUTPUT_ADDR_WIDTH,
            MAC_OUTPUT_DATA_WIDTH => MAC_OUTPUT_DATA_WIDTH,
            MAC_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            S_AXI_LITE_ACLK => S_AXI_LITE_ACLK,
            S_AXI_LITE_ARESETN => S_AXI_LITE_ARESETN,
            S_AXI_LITE_AWVALID => S_AXI_LITE_AWVALID,
            S_AXI_LITE_AWREADY => S_AXI_LITE_AWREADY,
            S_AXI_LITE_AWADDR => S_AXI_LITE_AWADDR,
            S_AXI_LITE_AWPROT => S_AXI_LITE_AWPROT,
            S_AXI_LITE_WVALID => S_AXI_LITE_WVALID,
            S_AXI_LITE_WREADY => S_AXI_LITE_WREADY,
            S_AXI_LITE_WDATA => S_AXI_LITE_WDATA,
            S_AXI_LITE_WSTRB => S_AXI_LITE_WSTRB,
            S_AXI_LITE_BVALID => S_AXI_LITE_BVALID,
            S_AXI_LITE_BREADY => S_AXI_LITE_BREADY,
            S_AXI_LITE_BRESP => S_AXI_LITE_BRESP,
            S_AXI_LITE_ARVALID => S_AXI_LITE_ARVALID,
            S_AXI_LITE_ARREADY => S_AXI_LITE_ARREADY,
            S_AXI_LITE_ARADDR => S_AXI_LITE_ARADDR,
            S_AXI_LITE_ARPROT => S_AXI_LITE_ARPROT,
            S_AXI_LITE_RVALID => S_AXI_LITE_RVALID,
            S_AXI_LITE_RREADY => S_AXI_LITE_RREADY,
            S_AXI_LITE_RDATA => S_AXI_LITE_RDATA,
            S_AXI_LITE_RRESP => S_AXI_LITE_RRESP,
            
            -- Configuration signals
            conv_idle => s_conv_idle,
            accelerator_controls_activation_bram => accelerator_controls_activation_bram,
            swap_activations => swap_activations,
            swap_filters => swap_filters,
            relu => s_relu,
            max_pooling => s_max_pooling,
            filter_w => s_filter_w,
            filter_h => s_filter_h,
            filter_c => s_filter_c,
            output_w => s_output_w,
            output_h => s_output_h,
            input_end_diff_fw => s_input_end_diff_fw,
            input_end_diff_fh => s_input_end_diff_fh,
            input_end_diff_fc => s_input_end_diff_fc,
            input_end_diff_ow => s_input_end_diff_ow,
            output_elements_per_channel => s_output_elements_per_channel,
            output_initial_offset => s_output_initial_offset,
            mac0_bias => s_mac0_bias,
            mac2_bias => s_mac2_bias,
            mac1_bias => s_mac1_bias,
            mac3_bias => s_mac3_bias,
            q_scale => s_q_scale,
            q_zero => s_q_zero,
            
            conv_complete => s_conv_complete
        );

    g_conv_accel: entity work.conv_accelerator
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
            relu => s_relu,
            max_pooling => s_max_pooling,
            filter_w => s_filter_w,
            filter_h => s_filter_h,
            filter_c => s_filter_c,
            output_w => s_output_w,
            output_h => s_output_h,
            input_end_diff_fw => s_input_end_diff_fw,
            input_end_diff_fh => s_input_end_diff_fh,
            input_end_diff_fc => s_input_end_diff_fc,
            input_end_diff_ow => s_input_end_diff_ow,
            output_elements_per_channel => s_output_elements_per_channel,
            output_initial_offset => s_output_initial_offset,
            mac0_bias => s_mac0_bias,
            mac1_bias => s_mac1_bias,
            mac2_bias => s_mac2_bias,
            mac3_bias => s_mac3_bias,
            q_scale => s_q_scale,
            q_zero => s_q_zero,

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

            BRAM_OUTPUT_addr => BRAM_OUTPUT_addr,
            BRAM_OUTPUT_din => BRAM_OUTPUT_din,
            BRAM_OUTPUT_dout => BRAM_OUTPUT_dout,
            BRAM_OUTPUT_en => BRAM_OUTPUT_en,
            BRAM_OUTPUT_we => BRAM_OUTPUT_we,
            BRAM_OUTPUT_rst => BRAM_OUTPUT_rst,
            BRAM_OUTPUT_clk => BRAM_OUTPUT_clk,

            TEST_s_index_gen_m_axis_tready => open,
            TEST_s_index_gen_m_axis_tdata_input_addr => open,
            TEST_s_index_gen_m_axis_tdata_filter_addr => open,
            TEST_s_index_gen_m_axis_tlast => open,
            TEST_s_index_gen_m_axis_tvalid => open,

            TEST_s_mac0_s_axis_tready => open,
            TEST_s_mac0_s_axis_tdata => open,
            TEST_s_mac0_s_axis_tlast => open,
            TEST_s_mac0_s_axis_tvalid => open,
            TEST_s_mac1_s_axis_tready => open,
            TEST_s_mac1_s_axis_tdata => open,
            TEST_s_mac1_s_axis_tlast => open,
            TEST_s_mac1_s_axis_tvalid => open,
            TEST_s_mac2_s_axis_tready => open,
            TEST_s_mac2_s_axis_tdata => open,
            TEST_s_mac2_s_axis_tlast => open,
            TEST_s_mac2_s_axis_tvalid => open,
            TEST_s_mac3_s_axis_tready => open,
            TEST_s_mac3_s_axis_tdata => open,
            TEST_s_mac3_s_axis_tlast => open,
            TEST_s_mac3_s_axis_tvalid => open,

            TEST_s_mac0_m_axis_tready => open,
            TEST_s_mac0_m_axis_tdata => open,
            TEST_s_mac0_m_axis_tlast => open,
            TEST_s_mac0_m_axis_tvalid => open,
            TEST_s_mac1_m_axis_tready => open,
            TEST_s_mac1_m_axis_tdata => open,
            TEST_s_mac1_m_axis_tlast => open,
            TEST_s_mac1_m_axis_tvalid => open,
            TEST_s_mac2_m_axis_tready => open,
            TEST_s_mac2_m_axis_tdata => open,
            TEST_s_mac2_m_axis_tlast => open,
            TEST_s_mac2_m_axis_tvalid => open,
            TEST_s_mac3_m_axis_tready => open,
            TEST_s_mac3_m_axis_tdata => open,
            TEST_s_mac3_m_axis_tlast => open,
            TEST_s_mac3_m_axis_tvalid => open,

            TEST_s_out_combiner_m_axis_tready => open,
            TEST_s_out_combiner_m_axis_tdata => open,
            TEST_s_out_combiner_m_axis_tlast => open,
            TEST_s_out_combiner_m_axis_tid => open,
            TEST_s_out_combiner_m_axis_tvalid => open,

            TEST_s_dequantization_m_axis_tready => open,
            TEST_s_dequantization_m_axis_tdata => open,
            TEST_s_dequantization_m_axis_tlast => open,
            TEST_s_dequantization_m_axis_tid => open,
            TEST_s_dequantization_m_axis_tvalid => open,

            conv_complete => s_conv_complete,
            conv_idle => s_conv_idle,
            rst => s_reset,
            clk => s_clk
        );


end Behavioral;

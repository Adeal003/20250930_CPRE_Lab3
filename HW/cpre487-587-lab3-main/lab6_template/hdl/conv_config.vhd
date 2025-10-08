----------------------------------------------------------------------------------
-- Convolutional Config Unit
--
-- Gregory Ling, 2024
-- Referencing https://github.com/ZipCPU/wb2axip/blob/master/rtl/easyaxil.v
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.math_real.all;

entity conv_config is
    generic(
        C_AXI_ADDR_WIDTH : integer := 8; -- 2^(C_AXI_ADDR_WIDTH-2) registers
        C_AXI_DATA_WIDTH : integer := 32; -- Fixed by AXI-lite spec
        DIM_WIDTH : integer := 12;
        INPUT_ADDR_WIDTH : integer := 17;
        FILTER_ADDR_WIDTH : integer := 10;
        OUTPUT_ADDR_WIDTH : integer := 17;
        MAC_OUTPUT_DATA_WIDTH : integer := 32;
        MAC_DATA_WIDTH : integer := 8
    );
    port(
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
        
        -- Configuration signals
        swap_activations : out std_logic;
        swap_filters : out std_logic;
        accelerator_controls_activation_bram : out std_logic;

        max_pooling : out std_logic;
        relu : out std_logic;
        filter_w : out std_logic_vector(DIM_WIDTH-1 downto 0);
        filter_h : out std_logic_vector(DIM_WIDTH-1 downto 0);
        filter_c : out std_logic_vector(DIM_WIDTH-1 downto 0);
        output_w : out std_logic_vector(DIM_WIDTH-1 downto 0);
        output_h : out std_logic_vector(DIM_WIDTH-1 downto 0);
        input_end_diff_fw : out std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        input_end_diff_fh : out std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        input_end_diff_fc : out std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        input_end_diff_ow : out std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        output_elements_per_channel : out std_logic_vector(OUTPUT_ADDR_WIDTH-1 downto 0);
        output_initial_offset : out std_logic_vector(OUTPUT_ADDR_WIDTH-1 downto 0);
        mac0_bias : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        mac2_bias : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        mac1_bias : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        mac3_bias : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        q_scale : out std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
        q_zero : out std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
        
        conv_complete : in std_logic;
        conv_idle : out std_logic
    );
end conv_config;

architecture Behavioral of conv_config is

    -- The least significant bit in the address we care about as we are using 32-bit registers
    constant ADDRLSB : integer := integer(ceil(log2(real(C_AXI_DATA_WIDTH/8))));

    signal axil_write_ready : std_logic;
    signal axil_write_response_valid : std_logic;
    signal axil_read_ready : std_logic;
    signal axil_read_valid : std_logic;
    signal axil_read_data : std_logic_vector(C_AXI_DATA_WIDTH-1 downto 0);
    signal axil_read_reg_index : std_logic_vector(C_AXI_ADDR_WIDTH-ADDRLSB-1 downto 0);
    signal axil_write_reg_index : std_logic_vector(C_AXI_ADDR_WIDTH-ADDRLSB-1 downto 0);

    -- Custom Peripheral Signals

    signal s_conv_idle : std_logic;
    signal s_relu : std_logic;
    signal s_max_pooling : std_logic;
    signal s_swap_activations : std_logic;
    signal s_swap_filters : std_logic;
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
    signal s_mac0_bias : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac2_bias : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac1_bias : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_mac3_bias : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_q_scale : std_logic_vector(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
    signal s_q_zero : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);

begin

    -- Write signaling

    process(S_AXI_LITE_ACLK) is
    begin
        if rising_edge(S_AXI_LITE_ACLK) then
            if (S_AXI_LITE_ARESETN = '0') then
                axil_write_ready <= '0';
                axil_write_response_valid <= '0';
            else
                axil_write_ready <= not axil_write_ready
                    and (S_AXI_LITE_AWVALID and S_AXI_LITE_WVALID)
                    and (not axil_write_response_valid or S_AXI_LITE_BREADY);
                axil_write_response_valid <= axil_write_ready or (not S_AXI_LITE_BREADY and axil_write_response_valid);
            end if;
        end if;
    end process;

    S_AXI_LITE_AWREADY <= axil_write_ready;
    S_AXI_LITE_WREADY <= axil_write_ready;
    S_AXI_LITE_BVALID <= axil_write_response_valid;
    S_AXI_LITE_BRESP <= "00";
    axil_write_reg_index <= S_AXI_LITE_AWADDR(C_AXI_ADDR_WIDTH-1 downto ADDRLSB);

    -- Read signaling

    S_AXI_LITE_ARREADY <= not axil_read_valid;
    axil_read_ready <= S_AXI_LITE_ARVALID and not axil_read_valid;
    process(S_AXI_LITE_ACLK) is
    begin
        if rising_edge(S_AXI_LITE_ACLK) then
            if (S_AXI_LITE_ARESETN = '0') then
                axil_read_valid <= '0';
            else
                axil_read_valid <= axil_read_ready or (not S_AXI_LITE_RREADY and axil_read_valid);
            end if;
        end if;
    end process;
    S_AXI_LITE_RVALID <= axil_read_valid;
    S_AXI_LITE_RDATA <= axil_read_data;
    S_AXI_LITE_RRESP <= "00";
    axil_read_reg_index <= S_AXI_LITE_ARADDR(C_AXI_ADDR_WIDTH-1 downto ADDRLSB);

    -- Register Write

    process(S_AXI_LITE_ACLK) is
    begin
        if rising_edge(S_AXI_LITE_ACLK) then
            if (S_AXI_LITE_ARESETN = '0') then
                s_conv_idle <= '1';
                s_max_pooling <= '0';
                s_swap_activations <= '0';
                s_swap_filters <= '0';
                s_filter_w <= (others => '0');
                s_filter_h <= (others => '0');
                s_filter_c <= (others => '0');
                s_output_w <= (others => '0');
                s_output_h <= (others => '0');
                s_input_end_diff_fw <= (others => '0');
                s_input_end_diff_fh <= (others => '0');
                s_input_end_diff_fc <= (others => '0');
                s_input_end_diff_ow <= (others => '0');
                s_output_elements_per_channel <= (others => '0');
                s_output_initial_offset <= (others => '0');
                s_mac0_bias <= (others => '0');
                s_mac1_bias <= (others => '0');
                s_mac2_bias <= (others => '0');
                s_mac3_bias <= (others => '0');
                s_q_scale <= (others => '0');
                s_q_zero <= (others => '0');

            elsif (axil_write_ready = '1') then
                case axil_write_reg_index is

                    -- EDIT REGISTER WRITE BEHAVIOR HERE
                    when "000000" =>
                        s_conv_idle <= S_AXI_LITE_WDATA(0);
                    when "000001" =>
                        s_relu <= S_AXI_LITE_WDATA(3);
                        s_max_pooling <= S_AXI_LITE_WDATA(2);
                        s_swap_activations <= S_AXI_LITE_WDATA(1);
                        s_swap_filters <= S_AXI_LITE_WDATA(0);
                    -- when "000010" =>
                    when "000011" =>
                        s_filter_w <= S_AXI_LITE_WDATA(DIM_WIDTH-1 downto 0);
                    when "000100" =>
                        s_filter_h <= S_AXI_LITE_WDATA(DIM_WIDTH-1 downto 0);
                    when "000101" =>
                        s_filter_c <= S_AXI_LITE_WDATA(DIM_WIDTH-1 downto 0);
                    when "000110" =>
                        s_output_w <= S_AXI_LITE_WDATA(DIM_WIDTH-1 downto 0);
                    when "000111" =>
                        s_output_h <= S_AXI_LITE_WDATA(DIM_WIDTH-1 downto 0);
                    when "001000" =>
                        s_input_end_diff_fw <= S_AXI_LITE_WDATA(INPUT_ADDR_WIDTH-1 downto 0);
                    when "001001" =>
                        s_input_end_diff_fh <= S_AXI_LITE_WDATA(INPUT_ADDR_WIDTH-1 downto 0);
                    when "001010" =>
                        s_input_end_diff_fc <= S_AXI_LITE_WDATA(INPUT_ADDR_WIDTH-1 downto 0);
                    when "001011" =>
                        s_input_end_diff_ow <= S_AXI_LITE_WDATA(INPUT_ADDR_WIDTH-1 downto 0);
                    when "001100" =>
                        s_output_elements_per_channel <= S_AXI_LITE_WDATA(OUTPUT_ADDR_WIDTH-1 downto 0);
                    when "001101" =>
                        s_output_initial_offset <= S_AXI_LITE_WDATA(OUTPUT_ADDR_WIDTH-1 downto 0);
                    when "001110" =>
                        s_mac0_bias <= S_AXI_LITE_WDATA(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
                    when "001111" =>
                        s_mac1_bias <= S_AXI_LITE_WDATA(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
                    when "010000" =>
                        s_mac2_bias <= S_AXI_LITE_WDATA(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
                    when "010001" =>
                        s_mac3_bias <= S_AXI_LITE_WDATA(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
                    when "010010" =>
                        s_q_scale <= S_AXI_LITE_WDATA(MAC_OUTPUT_DATA_WIDTH-1 downto 0);
                    when "010011" =>
                        s_q_zero <= S_AXI_LITE_WDATA(MAC_DATA_WIDTH-1 downto 0);
                    when others =>
                
                end case;
            elsif (conv_complete = '1' and s_conv_idle = '0') then
                s_conv_idle <= '1';
            end if;
        end if;
    end process;

    -- Register Read

    process(S_AXI_LITE_ACLK) is
    begin
        if rising_edge(S_AXI_LITE_ACLK) then
            axil_read_data <= (others => '0');
            if (axil_read_ready = '1') then
                case axil_read_reg_index is

                    -- EDIT REGISTER READ BEHAVIOR HERE
                    when "000000" =>
                        axil_read_data(0) <= s_conv_idle;
                    when "000001" =>
                        axil_read_data(3) <= s_relu;
                        axil_read_data(2) <= s_max_pooling;
                        axil_read_data(1) <= s_swap_activations;
                        axil_read_data(0) <= s_swap_filters;
                    -- when "000010" =>
                    when "000011" =>
                        axil_read_data(DIM_WIDTH-1 downto 0) <= s_filter_w;
                    when "000100" =>
                        axil_read_data(DIM_WIDTH-1 downto 0) <= s_filter_h;
                    when "000101" =>
                        axil_read_data(DIM_WIDTH-1 downto 0) <= s_filter_c;
                    when "000110" =>
                        axil_read_data(DIM_WIDTH-1 downto 0) <= s_output_w;
                    when "000111" =>
                        axil_read_data(DIM_WIDTH-1 downto 0) <= s_output_h;
                    when "001000" =>
                        axil_read_data(INPUT_ADDR_WIDTH-1 downto 0) <= s_input_end_diff_fw;
                    when "001001" =>
                        axil_read_data(INPUT_ADDR_WIDTH-1 downto 0) <= s_input_end_diff_fh;
                    when "001010" =>
                        axil_read_data(INPUT_ADDR_WIDTH-1 downto 0) <= s_input_end_diff_fc;
                    when "001011" =>
                        axil_read_data(INPUT_ADDR_WIDTH-1 downto 0) <= s_input_end_diff_ow;
                    when "001100" =>
                        axil_read_data(OUTPUT_ADDR_WIDTH-1 downto 0) <= s_output_elements_per_channel;
                    when "001101" =>
                        axil_read_data(OUTPUT_ADDR_WIDTH-1 downto 0) <= s_output_initial_offset;
                    when "001110" =>
                        axil_read_data(MAC_OUTPUT_DATA_WIDTH-1 downto 0) <= s_mac0_bias;
                    when "001111" =>
                        axil_read_data(MAC_OUTPUT_DATA_WIDTH-1 downto 0) <= s_mac1_bias;
                    when "010000" =>
                        axil_read_data(MAC_OUTPUT_DATA_WIDTH-1 downto 0) <= s_mac2_bias;
                    when "010001" =>
                        axil_read_data(MAC_OUTPUT_DATA_WIDTH-1 downto 0) <= s_mac3_bias;
                    when "010010" =>
                        axil_read_data(MAC_OUTPUT_DATA_WIDTH-1 downto 0) <= s_q_scale;
                    when "010011" =>
                        axil_read_data(MAC_DATA_WIDTH-1 downto 0) <= s_q_zero;
                    when others =>

                end case;
            else
                axil_read_data <= (others => '0');
            end if;
        end if;
    end process;

    -- Custom Peripheral Logic

    relu <= s_relu;
    max_pooling <= s_max_pooling;
    accelerator_controls_activation_bram <= not s_conv_idle;
    swap_activations <= s_swap_activations;
    swap_filters <= s_swap_filters;
    filter_w <= s_filter_w;
    filter_h <= s_filter_h;
    filter_c <= s_filter_c;
    output_w <= s_output_w;
    output_h <= s_output_h;
    input_end_diff_fw <= s_input_end_diff_fw;
    input_end_diff_fh <= s_input_end_diff_fh;
    input_end_diff_fc <= s_input_end_diff_fc;
    input_end_diff_ow <= s_input_end_diff_ow;
    output_elements_per_channel <= s_output_elements_per_channel;
    output_initial_offset <= s_output_initial_offset;
    mac0_bias <= s_mac0_bias;
    mac2_bias <= s_mac2_bias;
    mac1_bias <= s_mac1_bias;
    mac3_bias <= s_mac3_bias;
    q_scale <= s_q_scale;
    q_zero <= s_q_zero;

    conv_idle <= s_conv_idle;

end Behavioral;

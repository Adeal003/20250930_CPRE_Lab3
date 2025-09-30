----------------------------------------------------------------------------------
-- MAC Stream Provideer
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity mac_stream_provider is
    generic(
        INPUT_ADDR_WIDTH : integer := 32;
        FILTER_ADDR_WIDTH : integer := 32;
        INPUT_BRAM_ADDR_WIDTH : integer := 32;
        FILTER_BRAM_ADDR_WIDTH : integer := 32;
        BRAM_DATA_WIDTH : integer := 32;
        MAC_DATA_WIDTH : integer := 32
    );
    port(
        S_AXIS_TREADY : out std_logic;
        S_AXIS_TDATA_input_addr : in std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
        S_AXIS_TDATA_filter_addr : in std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
        S_AXIS_TLAST : in std_logic;
        S_AXIS_TVALID : in std_logic;
        
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

        M_AXIS_MAC0_TREADY : in  std_logic;
        M_AXIS_MAC0_TDATA  : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        M_AXIS_MAC0_TLAST  : out std_logic;
        M_AXIS_MAC0_TVALID : out std_logic;

        M_AXIS_MAC1_TREADY : in  std_logic;
        M_AXIS_MAC1_TDATA  : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        M_AXIS_MAC1_TLAST  : out std_logic;
        M_AXIS_MAC1_TVALID : out std_logic;

        M_AXIS_MAC2_TREADY : in  std_logic;
        M_AXIS_MAC2_TDATA  : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        M_AXIS_MAC2_TLAST  : out std_logic;
        M_AXIS_MAC2_TVALID : out std_logic;

        M_AXIS_MAC3_TREADY : in  std_logic;
        M_AXIS_MAC3_TDATA  : out std_logic_vector(MAC_DATA_WIDTH*2-1 downto 0);
        M_AXIS_MAC3_TLAST  : out std_logic;
        M_AXIS_MAC3_TVALID : out std_logic;

        clk : in std_logic;
        rst : in std_logic
    );
end mac_stream_provider;


architecture behavioral of mac_stream_provider is

    signal s_reg0_s_axis_tready : std_logic;
    signal s_reg0_s_axis_tdata : std_logic_vector(INPUT_ADDR_WIDTH+FILTER_ADDR_WIDTH-1 downto 0);
    signal s_reg0_m_axis_tdata : std_logic_vector(INPUT_ADDR_WIDTH+FILTER_ADDR_WIDTH-1 downto 0);
    signal s_reg0_m_axis_tdata_input : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_reg0_m_axis_tdata_filter : std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
    signal s_reg0_m_axis_tready : std_logic;
    signal s_reg0_m_axis_tvalid : std_logic;

    signal s_reg1_s_axis_tready : std_logic;
    signal s_reg1_s_axis_input_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_s_axis_filter0_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_s_axis_filter1_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_s_axis_filter2_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_s_axis_filter3_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_s_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*5-1 downto 0);
    signal s_reg1_s_axis_tlast : std_logic;
    signal s_reg1_s_axis_tvalid : std_logic;

    signal s_reg1_m_axis_tready : std_logic;
    signal s_reg1_m_axis_input_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_m_axis_filter0_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_m_axis_filter1_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_m_axis_filter2_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_m_axis_filter3_data : std_logic_vector(MAC_DATA_WIDTH-1 downto 0);
    signal s_reg1_m_axis_tdata : std_logic_vector(MAC_DATA_WIDTH*5-1 downto 0);

    signal s_reg1_m_axis_tlast : std_logic;
    signal s_reg1_m_axis_tvalid : std_logic;

    -- Buffer data coming from BRAM units to help meet timing constraints

    signal s_bram_en : std_logic;
    signal s_bram_input_addr : std_logic_vector(INPUT_ADDR_WIDTH-1 downto 0);
    signal s_bram_filter_addr : std_logic_vector(FILTER_ADDR_WIDTH-1 downto 0);
    signal s_backlogged : std_logic;

begin

    S_AXIS_TREADY <= s_reg0_s_axis_tready;

    process(clk) begin
        if rising_edge(clk) then
            s_backlogged <= s_reg1_s_axis_tvalid and not s_reg1_s_axis_tready;
        end if;
    end process;

    s_reg0_s_axis_tdata(INPUT_ADDR_WIDTH+FILTER_ADDR_WIDTH-1 downto FILTER_ADDR_WIDTH) <= S_AXIS_TDATA_input_addr;
    s_reg0_s_axis_tdata(FILTER_ADDR_WIDTH-1 downto 0) <= S_AXIS_TDATA_filter_addr;
    -- BRAM units read on the clock edge, this delays the stream control signals appropriately
    g_register0: entity work.axis_register_slice
        generic map(
            C_DATA_WIDTH => INPUT_ADDR_WIDTH+FILTER_ADDR_WIDTH,
            C_TID_WIDTH => 1
        )
        port map(
            S_AXIS_TREADY => s_reg0_s_axis_tready,
            S_AXIS_TDATA => s_reg0_s_axis_tdata,
            S_AXIS_TLAST => S_AXIS_TLAST,
            S_AXIS_TID => (others => '0'),
            S_AXIS_TVALID => S_AXIS_TVALID,

            M_AXIS_TREADY => s_reg0_m_axis_tready,
            M_AXIS_TDATA => s_reg0_m_axis_tdata,
            M_AXIS_TLAST => s_reg1_s_axis_tlast,
            M_AXIS_TID => open,
            M_AXIS_TVALID => s_reg0_m_axis_tvalid,
            
            rst => rst,
            clk => clk
        );
    s_reg0_m_axis_tdata_input <= s_reg0_m_axis_tdata(INPUT_ADDR_WIDTH+FILTER_ADDR_WIDTH-1 downto FILTER_ADDR_WIDTH);
    s_reg0_m_axis_tdata_filter <= s_reg0_m_axis_tdata(FILTER_ADDR_WIDTH-1 downto 0);

    s_reg1_s_axis_tvalid <= s_reg0_m_axis_tvalid and not s_backlogged;
    s_reg0_m_axis_tready <= s_reg1_s_axis_tready and not s_backlogged;

    -- BRAM units should be enabled when reg0 latches
    s_bram_en <= S_AXIS_TVALID and s_reg0_s_axis_tready;

    -- Be careful of becoming backlogged
    s_bram_input_addr <= s_reg0_m_axis_tdata_input when s_backlogged = '1' else S_AXIS_TDATA_input_addr;
    s_bram_filter_addr <= s_reg0_m_axis_tdata_filter when s_backlogged = '1' else S_AXIS_TDATA_filter_addr;

    g_input_fetcher: entity work.bram_slice_fetcher
        generic map(
            ADDR_WIDTH => INPUT_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            BRAM_ADDR_WIDTH => INPUT_BRAM_ADDR_WIDTH,
            OUT_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            BRAM_addr => BRAM_INPUT_addr,
            BRAM_din => BRAM_INPUT_din,
            BRAM_dout => BRAM_INPUT_dout,
            BRAM_en => BRAM_INPUT_en,
            BRAM_we => BRAM_INPUT_we,
            BRAM_rst => BRAM_INPUT_rst,
            BRAM_clk => BRAM_INPUT_clk,

            addr => s_bram_input_addr,
            data => s_reg1_s_axis_input_data,

            en => s_bram_en,
            rst => rst,
            clk => clk
        );

    g_filter0_fetcher: entity work.bram_slice_fetcher
        generic map(
            ADDR_WIDTH => FILTER_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            BRAM_ADDR_WIDTH => FILTER_BRAM_ADDR_WIDTH,
            OUT_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            BRAM_addr => BRAM_FILTER0_addr,
            BRAM_din => BRAM_FILTER0_din,
            BRAM_dout => BRAM_FILTER0_dout,
            BRAM_en => BRAM_FILTER0_en,
            BRAM_we => BRAM_FILTER0_we,
            BRAM_rst => BRAM_FILTER0_rst,
            BRAM_clk => BRAM_FILTER0_clk,

            addr => s_bram_filter_addr,
            data => s_reg1_s_axis_filter0_data,

            en => s_bram_en,
            rst => rst,
            clk => clk
        );

    g_filter1_fetcher: entity work.bram_slice_fetcher
        generic map(
            ADDR_WIDTH => FILTER_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            BRAM_ADDR_WIDTH => FILTER_BRAM_ADDR_WIDTH,
            OUT_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            BRAM_addr => BRAM_FILTER1_addr,
            BRAM_din => BRAM_FILTER1_din,
            BRAM_dout => BRAM_FILTER1_dout,
            BRAM_en => BRAM_FILTER1_en,
            BRAM_we => BRAM_FILTER1_we,
            BRAM_rst => BRAM_FILTER1_rst,
            BRAM_clk => BRAM_FILTER1_clk,

            addr => s_bram_filter_addr,
            data => s_reg1_s_axis_filter1_data,

            en => s_bram_en,
            rst => rst,
            clk => clk
        );

    g_filter2_fetcher: entity work.bram_slice_fetcher
        generic map(
            ADDR_WIDTH => FILTER_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            BRAM_ADDR_WIDTH => FILTER_BRAM_ADDR_WIDTH,
            OUT_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            BRAM_addr => BRAM_FILTER2_addr,
            BRAM_din => BRAM_FILTER2_din,
            BRAM_dout => BRAM_FILTER2_dout,
            BRAM_en => BRAM_FILTER2_en,
            BRAM_we => BRAM_FILTER2_we,
            BRAM_rst => BRAM_FILTER2_rst,
            BRAM_clk => BRAM_FILTER2_clk,

            addr => s_bram_filter_addr,
            data => s_reg1_s_axis_filter2_data,

            en => s_bram_en,
            rst => rst,
            clk => clk
        );

    g_filter3_fetcher: entity work.bram_slice_fetcher
        generic map(
            ADDR_WIDTH => FILTER_ADDR_WIDTH,
            BRAM_DATA_WIDTH => BRAM_DATA_WIDTH,
            BRAM_ADDR_WIDTH => FILTER_BRAM_ADDR_WIDTH,
            OUT_DATA_WIDTH => MAC_DATA_WIDTH
        )
        port map(
            BRAM_addr => BRAM_FILTER3_addr,
            BRAM_din => BRAM_FILTER3_din,
            BRAM_dout => BRAM_FILTER3_dout,
            BRAM_en => BRAM_FILTER3_en,
            BRAM_we => BRAM_FILTER3_we,
            BRAM_rst => BRAM_FILTER3_rst,
            BRAM_clk => BRAM_FILTER3_clk,

            addr => s_bram_filter_addr,
            data => s_reg1_s_axis_filter3_data,

            en => s_bram_en,
            rst => rst,
            clk => clk
        );

    -- And buffer the data once coming out of the BRAM blocks to help with the critical timing path
    s_reg1_s_axis_tdata(MAC_DATA_WIDTH*5-1 downto MAC_DATA_WIDTH*4) <= s_reg1_s_axis_input_data;
    s_reg1_s_axis_tdata(MAC_DATA_WIDTH*4-1 downto MAC_DATA_WIDTH*3) <= s_reg1_s_axis_filter0_data;
    s_reg1_s_axis_tdata(MAC_DATA_WIDTH*3-1 downto MAC_DATA_WIDTH*2) <= s_reg1_s_axis_filter1_data;
    s_reg1_s_axis_tdata(MAC_DATA_WIDTH*2-1 downto MAC_DATA_WIDTH*1) <= s_reg1_s_axis_filter2_data;
    s_reg1_s_axis_tdata(MAC_DATA_WIDTH*1-1 downto MAC_DATA_WIDTH*0) <= s_reg1_s_axis_filter3_data;
    g_register1: entity work.axis_register_slice
        generic map(
            C_DATA_WIDTH => MAC_DATA_WIDTH*5,
            C_TID_WIDTH => 1
        )
        port map(
            S_AXIS_TREADY => s_reg1_s_axis_tready,
            S_AXIS_TDATA => s_reg1_s_axis_tdata,
            S_AXIS_TLAST => s_reg1_s_axis_tlast,
            S_AXIS_TID => (others => '0'),
            S_AXIS_TVALID => s_reg1_s_axis_tvalid,

            M_AXIS_TREADY => s_reg1_m_axis_tready,
            M_AXIS_TDATA => s_reg1_m_axis_tdata,
            M_AXIS_TLAST => s_reg1_m_axis_tlast,
            M_AXIS_TID => open,
            M_AXIS_TVALID => s_reg1_m_axis_tvalid,

            rst => rst,
            clk => clk
        );
    s_reg1_m_axis_input_data <= s_reg1_m_axis_tdata(MAC_DATA_WIDTH*5-1 downto MAC_DATA_WIDTH*4);
    s_reg1_m_axis_filter0_data <= s_reg1_m_axis_tdata(MAC_DATA_WIDTH*4-1 downto MAC_DATA_WIDTH*3);
    s_reg1_m_axis_filter1_data <= s_reg1_m_axis_tdata(MAC_DATA_WIDTH*3-1 downto MAC_DATA_WIDTH*2);
    s_reg1_m_axis_filter2_data <= s_reg1_m_axis_tdata(MAC_DATA_WIDTH*2-1 downto MAC_DATA_WIDTH*1);
    s_reg1_m_axis_filter3_data <= s_reg1_m_axis_tdata(MAC_DATA_WIDTH*1-1 downto MAC_DATA_WIDTH*0);
  
    s_reg1_m_axis_tready <= M_AXIS_MAC0_TREADY and M_AXIS_MAC1_TREADY and M_AXIS_MAC2_TREADY and M_AXIS_MAC3_TREADY;

    -- Only transfer when all macs are ready at the same time
    M_AXIS_MAC0_TVALID <= s_reg1_m_axis_tvalid and s_reg1_m_axis_tready;
    M_AXIS_MAC1_TVALID <= s_reg1_m_axis_tvalid and s_reg1_m_axis_tready;
    M_AXIS_MAC2_TVALID <= s_reg1_m_axis_tvalid and s_reg1_m_axis_tready;
    M_AXIS_MAC3_TVALID <= s_reg1_m_axis_tvalid and s_reg1_m_axis_tready;

    M_AXIS_MAC0_TDATA <= s_reg1_m_axis_input_data & s_reg1_m_axis_filter0_data;
    M_AXIS_MAC1_TDATA <= s_reg1_m_axis_input_data & s_reg1_m_axis_filter1_data;
    M_AXIS_MAC2_TDATA <= s_reg1_m_axis_input_data & s_reg1_m_axis_filter2_data;
    M_AXIS_MAC3_TDATA <= s_reg1_m_axis_input_data & s_reg1_m_axis_filter3_data;

    M_AXIS_MAC0_TLAST <= s_reg1_m_axis_tlast;
    M_AXIS_MAC1_TLAST <= s_reg1_m_axis_tlast;
    M_AXIS_MAC2_TLAST <= s_reg1_m_axis_tlast;
    M_AXIS_MAC3_TLAST <= s_reg1_m_axis_tlast;

end architecture behavioral;

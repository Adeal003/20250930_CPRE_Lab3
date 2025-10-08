----------------------------------------------------------------------------------
-- BRAM Read Delay
-- Insert one read pipeline stage (increases AXI BRAM Controller Read Latency by 1)
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity bram_read_delay is
    generic(
        ADDR_WIDTH : integer := 32;
        DATA_WIDTH : integer := 32
    );
    port(
        M_BRAM_addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        M_BRAM_clk : in std_logic;
        M_BRAM_din : in std_logic_vector(DATA_WIDTH-1 downto 0);
        M_BRAM_dout : out std_logic_vector(DATA_WIDTH-1 downto 0);
        M_BRAM_en : in std_logic;
        M_BRAM_we : in std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        M_BRAM_rst : in std_logic;

        S_BRAM_addr : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        S_BRAM_clk : out std_logic;
        S_BRAM_din : out std_logic_vector(DATA_WIDTH-1 downto 0);
        S_BRAM_dout : in std_logic_vector(DATA_WIDTH-1 downto 0);
        S_BRAM_en : out std_logic;
        S_BRAM_we : out std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        S_BRAM_rst : out std_logic
    );
end bram_read_delay;

architecture Behavioral of bram_read_delay is

    attribute X_INTERFACE_INFO : STRING;
    attribute X_INTERFACE_MODE : STRING;

    attribute X_INTERFACE_INFO of M_BRAM_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM ADDR";
    attribute X_INTERFACE_INFO of M_BRAM_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM CLK";
    attribute X_INTERFACE_INFO of M_BRAM_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM DIN";
    attribute X_INTERFACE_INFO of M_BRAM_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM DOUT";
    attribute X_INTERFACE_INFO of M_BRAM_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM EN";
    attribute X_INTERFACE_INFO of M_BRAM_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM WE";
    attribute X_INTERFACE_INFO of M_BRAM_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM RST";

    attribute X_INTERFACE_INFO of S_BRAM_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM ADDR";
    attribute X_INTERFACE_INFO of S_BRAM_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM CLK";
    attribute X_INTERFACE_INFO of S_BRAM_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM DIN";
    attribute X_INTERFACE_INFO of S_BRAM_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM DOUT";
    attribute X_INTERFACE_INFO of S_BRAM_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM EN";
    attribute X_INTERFACE_INFO of S_BRAM_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM WE";
    attribute X_INTERFACE_INFO of S_BRAM_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM RST";

    attribute X_INTERFACE_MODE of S_BRAM_addr:  SIGNAL is "Master";

    signal s_delayed_dout : std_logic_vector(DATA_WIDTH-1 downto 0);
begin

    process(M_BRAM_clk) begin
        if rising_edge(M_BRAM_clk) then
            s_delayed_dout <= S_BRAM_dout;
        end if;
    end process;

    M_BRAM_dout <= s_delayed_dout;
    S_BRAM_addr <= M_BRAM_addr;
    S_BRAM_din  <= M_BRAM_din;
    S_BRAM_en   <= M_BRAM_en;
    S_BRAM_we   <= M_BRAM_we;
    S_BRAM_rst  <= M_BRAM_rst;
    S_BRAM_clk  <= M_BRAM_clk;

end Behavioral;

----------------------------------------------------------------------------------
-- BRAM connect
-- Connect a pair of BRAM modules into a larger BRAM module, adding one address
-- bit. These can be connected to form a tree if needed.
-- Configure the BRAM module to Stand Alone mode.
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------


library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.MATH_REAL.ALL;

entity bram_connect is
    generic(
        M_ADDR_WIDTH : integer := 32;
        ADDR_WIDTH : integer := 32;
        DATA_WIDTH : integer := 64
    );
    port(
        M_BRAM0_addr : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        M_BRAM0_clk : out std_logic;
        M_BRAM0_din : out std_logic_vector(DATA_WIDTH-1 downto 0);
        M_BRAM0_dout : in std_logic_vector(DATA_WIDTH-1 downto 0);
        M_BRAM0_en : out std_logic; 
        M_BRAM0_we : out std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        M_BRAM0_rst : out std_logic;

        M_BRAM1_addr : out std_logic_vector(ADDR_WIDTH-1 downto 0);
        M_BRAM1_clk : out std_logic;
        M_BRAM1_din : out std_logic_vector(DATA_WIDTH-1 downto 0);
        M_BRAM1_dout : in std_logic_vector(DATA_WIDTH-1 downto 0);
        M_BRAM1_en : out std_logic;
        M_BRAM1_we : out std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        M_BRAM1_rst : out std_logic;

        S_BRAM_addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        S_BRAM_clk : in std_logic;
        S_BRAM_din : in std_logic_vector(DATA_WIDTH-1 downto 0);
        S_BRAM_dout : out std_logic_vector(DATA_WIDTH-1 downto 0);
        S_BRAM_en : in std_logic;
        S_BRAM_we : in std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        S_BRAM_rst : in std_logic
    );
end bram_connect;

architecture Behavioral of bram_connect is

    attribute X_INTERFACE_INFO : STRING;
    attribute X_INTERFACE_MODE : STRING;
    attribute X_INTERFACE_PARAMETER : STRING;

    attribute X_INTERFACE_INFO of M_BRAM0_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 ADDR";
    attribute X_INTERFACE_INFO of M_BRAM0_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 CLK";
    attribute X_INTERFACE_INFO of M_BRAM0_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 DIN";
    attribute X_INTERFACE_INFO of M_BRAM0_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 DOUT";
    attribute X_INTERFACE_INFO of M_BRAM0_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 EN";
    attribute X_INTERFACE_INFO of M_BRAM0_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 WE";
    attribute X_INTERFACE_INFO of M_BRAM0_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM0 RST";

    attribute X_INTERFACE_INFO of M_BRAM1_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 ADDR";
    attribute X_INTERFACE_INFO of M_BRAM1_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 CLK";
    attribute X_INTERFACE_INFO of M_BRAM1_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 DIN";
    attribute X_INTERFACE_INFO of M_BRAM1_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 DOUT";
    attribute X_INTERFACE_INFO of M_BRAM1_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 EN";
    attribute X_INTERFACE_INFO of M_BRAM1_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 WE";
    attribute X_INTERFACE_INFO of M_BRAM1_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 M_BRAM1 RST";

    attribute X_INTERFACE_INFO of S_BRAM_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM ADDR";
    attribute X_INTERFACE_INFO of S_BRAM_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM CLK";
    attribute X_INTERFACE_INFO of S_BRAM_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM DIN";
    attribute X_INTERFACE_INFO of S_BRAM_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM DOUT";
    attribute X_INTERFACE_INFO of S_BRAM_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM EN";
    attribute X_INTERFACE_INFO of S_BRAM_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM WE";
    attribute X_INTERFACE_INFO of S_BRAM_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 S_BRAM RST";

    attribute X_INTERFACE_MODE of M_BRAM0_addr:  SIGNAL is "Master"; 
    attribute X_INTERFACE_MODE of M_BRAM1_addr:  SIGNAL is "Master"; 

    constant READ_DELAY : integer := 1; -- MUST match the BRAM block summary
    signal s_read_delay : std_logic_vector(READ_DELAY-1 downto 0);
begin

    process(S_BRAM_clk)
    begin
        if rising_edge(S_BRAM_clk) then
            s_read_delay <= S_BRAM_addr(M_ADDR_WIDTH downto M_ADDR_WIDTH);-- & s_read_delay(READ_DELAY-1 downto 1);
        end if;
    end process;

    S_BRAM_dout <= M_BRAM0_dout when (s_read_delay(0) = '0') else M_BRAM1_dout;
    M_BRAM0_addr <= S_BRAM_addr;
    M_BRAM1_addr <= S_BRAM_addr;
    M_BRAM0_clk  <= S_BRAM_clk;
    M_BRAM1_clk  <= S_BRAM_clk;
    M_BRAM0_din  <= S_BRAM_din;
    M_BRAM1_din  <= S_BRAM_din;
    M_BRAM0_en   <= S_BRAM_en and not S_BRAM_addr(M_ADDR_WIDTH);
    M_BRAM1_en   <= S_BRAM_en and S_BRAM_addr(M_ADDR_WIDTH);
    M_BRAM0_we   <= S_BRAM_we;
    M_BRAM1_we   <= S_BRAM_we;
    M_BRAM0_rst  <= S_BRAM_rst;
    M_BRAM1_rst  <= S_BRAM_rst;
    
end Behavioral;

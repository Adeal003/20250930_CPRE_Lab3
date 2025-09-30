----------------------------------------------------------------------------------
-- BRAM Mux
-- Switch between two controllers of a BRAM module port.
-- Configure the BRAM module to Stand Alone mode.
-- 
-- Gregory Ling, 2024
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity bram_mux is
    generic(
        ADDR_DEPTH : integer := 13;
        DATA_WIDTH : integer := 64
    );
    port(
        BRAM_PORT0_addr : in std_logic_vector(ADDR_DEPTH-1 downto 0);
        BRAM_PORT0_clk : in std_logic;
        BRAM_PORT0_din : in std_logic_vector(DATA_WIDTH-1 downto 0);
        BRAM_PORT0_dout : out std_logic_vector(DATA_WIDTH-1 downto 0);
        BRAM_PORT0_en : in std_logic;
        BRAM_PORT0_we : in std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        BRAM_PORT0_rst : in std_logic;

        BRAM_PORT1_addr : in std_logic_vector(ADDR_DEPTH-1 downto 0);
        BRAM_PORT1_clk : in std_logic;
        BRAM_PORT1_din : in std_logic_vector(DATA_WIDTH-1 downto 0);
        BRAM_PORT1_dout : out std_logic_vector(DATA_WIDTH-1 downto 0);
        BRAM_PORT1_en : in std_logic;
        BRAM_PORT1_we : in std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        BRAM_PORT1_rst : in std_logic;

        sel : in std_logic;

        BRAM_PORT_addr : out std_logic_vector(ADDR_DEPTH-1 downto 0);
        BRAM_PORT_clk : out std_logic;
        BRAM_PORT_din : out std_logic_vector(DATA_WIDTH-1 downto 0);
        BRAM_PORT_dout : in std_logic_vector(DATA_WIDTH-1 downto 0);
        BRAM_PORT_en : out std_logic;
        BRAM_PORT_we : out std_logic_vector((DATA_WIDTH/8)-1 downto 0);
        BRAM_PORT_rst : out std_logic
    );
end bram_mux;

architecture Behavioral of bram_mux is

    attribute X_INTERFACE_INFO : STRING;
    attribute X_INTERFACE_MODE : STRING;

    attribute X_INTERFACE_INFO of BRAM_PORT0_addr: SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 ADDR";
    attribute X_INTERFACE_INFO of BRAM_PORT0_clk:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 CLK";
    attribute X_INTERFACE_INFO of BRAM_PORT0_din:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 DIN";
    attribute X_INTERFACE_INFO of BRAM_PORT0_dout: SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 DOUT";
    attribute X_INTERFACE_INFO of BRAM_PORT0_en:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 EN";
    attribute X_INTERFACE_INFO of BRAM_PORT0_we:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 WE";
    attribute X_INTERFACE_INFO of BRAM_PORT0_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT0 RST";

    attribute X_INTERFACE_INFO of BRAM_PORT1_addr: SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 ADDR";
    attribute X_INTERFACE_INFO of BRAM_PORT1_clk:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 CLK";
    attribute X_INTERFACE_INFO of BRAM_PORT1_din:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 DIN";
    attribute X_INTERFACE_INFO of BRAM_PORT1_dout: SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 DOUT";
    attribute X_INTERFACE_INFO of BRAM_PORT1_en:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 EN";
    attribute X_INTERFACE_INFO of BRAM_PORT1_we:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 WE";
    attribute X_INTERFACE_INFO of BRAM_PORT1_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT1 RST";
    
    attribute X_INTERFACE_INFO of BRAM_PORT_addr:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT ADDR";
    attribute X_INTERFACE_INFO of BRAM_PORT_clk:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT CLK";
    attribute X_INTERFACE_INFO of BRAM_PORT_din:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT DIN";
    attribute X_INTERFACE_INFO of BRAM_PORT_dout:  SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT DOUT";
    attribute X_INTERFACE_INFO of BRAM_PORT_en:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT EN";
    attribute X_INTERFACE_INFO of BRAM_PORT_we:    SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT WE";
    attribute X_INTERFACE_INFO of BRAM_PORT_rst:   SIGNAL is "xilinx.com:interface:bram_rtl:1.0 BRAM_PORT RST";
    attribute X_INTERFACE_MODE of BRAM_PORT_addr:  SIGNAL is "Master"; 
begin

    BRAM_PORT_addr <= BRAM_PORT0_addr when (sel = '0') else BRAM_PORT1_addr;
    BRAM_PORT_clk <= BRAM_PORT0_clk;
    BRAM_PORT_din <= BRAM_PORT0_din when (sel = '0') else BRAM_PORT1_din;
    BRAM_PORT_en <= BRAM_PORT0_en when (sel = '0') else BRAM_PORT1_en;
    BRAM_PORT_we <= BRAM_PORT0_we when (sel = '0') else BRAM_PORT1_we;
    BRAM_PORT_rst <= BRAM_PORT0_rst when (sel = '0') else BRAM_PORT1_rst;
    BRAM_PORT0_dout <= BRAM_PORT_dout when (sel = '0') else (others => '0');
    BRAM_PORT1_dout <= (others => '0') when (sel = '0') else BRAM_PORT_dout;
end Behavioral;

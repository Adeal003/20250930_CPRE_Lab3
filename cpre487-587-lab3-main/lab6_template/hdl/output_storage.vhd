----------------------------------------------------------------------------------
-- Output Storage Unit
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity output_storage is
    generic(
        DATA_WIDTH : integer := 32;
        BRAM_DATA_WIDTH : integer := 32;
        ADDR_WIDTH : integer := 32;
        BRAM_ADDR_WIDTH : integer := 32;
        DIM_WIDTH : integer := 8;
        C_TID_WIDTH : integer := 1
    );
    port(
        S_AXIS_TREADY : out std_logic;
        S_AXIS_TDATA  : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        S_AXIS_TLAST  : in  std_logic;
        S_AXIS_TID    : in  std_logic_vector(C_TID_WIDTH-1 downto 0);
        S_AXIS_TVALID : in  std_logic;

        BRAM_addr : out std_logic_vector(32-1 downto 0);
        BRAM_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_en : out std_logic;
        BRAM_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_rst : out std_logic;
        BRAM_clk : out std_logic;

        max_pooling : in std_logic;
        elements_per_channel : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        output_w : in std_logic_vector(DIM_WIDTH-1 downto 0);
        output_h : in std_logic_vector(DIM_WIDTH-1 downto 0);
        initial_offset : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        
        conv_complete : out std_logic;
        conv_idle : in std_logic;
        clk : in std_logic;
        rst : in std_logic
    );
end output_storage;

architecture Behavioral of output_storage is

-- TODO

begin

-- TODO

end Behavioral;

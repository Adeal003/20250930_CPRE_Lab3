----------------------------------------------------------------------------------
-- Convolutional MAC Unit
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity conv_mac is
  generic(
      C_DATA_WIDTH : integer := 32;
      C_OUTPUT_DATA_WIDTH : integer := 32
    );
    port (  
        S_AXIS_TREADY : out std_logic;
        S_AXIS_TDATA  : in  std_logic_vector(C_DATA_WIDTH*2-1 downto 0);
        S_AXIS_TLAST  : in  std_logic;
        S_AXIS_TVALID : in  std_logic;

        bias : in std_logic_vector(C_OUTPUT_DATA_WIDTH-1 downto 0);

        M_AXIS_TREADY : in  std_logic;
        M_AXIS_TDATA  : out std_logic_vector(C_OUTPUT_DATA_WIDTH-1 downto 0);
        M_AXIS_TLAST  : out std_logic;
        M_AXIS_TVALID : out std_logic;

        rst : in std_logic;
        clk : in std_logic
    );

end conv_mac;


architecture behavioral of conv_mac is

-- TODO

begin

-- TODO

end architecture behavioral;

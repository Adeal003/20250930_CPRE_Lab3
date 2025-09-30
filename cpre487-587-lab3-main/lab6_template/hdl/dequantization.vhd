----------------------------------------------------------------------------------
-- Dequantization Unit
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity dequantization is
    generic(
        C_DATA_WIDTH : integer := 32;
        C_TID_WIDTH : integer := 1;
        C_OUT_WIDTH : integer := 8
    );
    port(
        S_AXIS_TREADY : out std_logic;
        S_AXIS_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS_TLAST  : in  std_logic;
        S_AXIS_TID    : in  std_logic_vector(C_TID_WIDTH-1 downto 0);
        S_AXIS_TVALID : in  std_logic;

        relu : in std_logic;
        q_scale : in std_logic_vector(C_DATA_WIDTH-1 downto 0);
        q_zero : in std_logic_vector(C_OUT_WIDTH-1 downto 0);

        M_AXIS_TREADY : in  std_logic;
        M_AXIS_TDATA  : out std_logic_vector(C_OUT_WIDTH-1 downto 0);
        M_AXIS_TLAST  : out std_logic;
        M_AXIS_TID    : out std_logic_vector(C_TID_WIDTH-1 downto 0);
        M_AXIS_TVALID : out std_logic;

        clk : in std_logic;
        rst : in std_logic
    );
end dequantization;

architecture Behavioral of dequantization is

-- TODO

begin

-- TODO

end Behavioral;

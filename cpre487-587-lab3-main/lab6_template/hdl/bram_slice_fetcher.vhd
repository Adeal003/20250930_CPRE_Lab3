----------------------------------------------------------------------------------
-- BRAM Slice Fetcher
-- Slice an N-bit value out of a M-bit word from the BRAM.
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity bram_slice_fetcher is
    generic(
        ADDR_WIDTH : integer := 32;
        BRAM_DATA_WIDTH : integer := 32;
        BRAM_ADDR_WIDTH : integer := 32;
        OUT_DATA_WIDTH : integer := 8 -- supports sub-byte indexing, must be power of 2 and less than BRAM_DATA_WIDTH
    );
    port(
        BRAM_addr : out std_logic_vector(32-1 downto 0); -- BRAM is word-addressed
        BRAM_din : out std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_dout : in std_logic_vector(BRAM_DATA_WIDTH-1 downto 0);
        BRAM_en : out std_logic;
        BRAM_we : out std_logic_vector((BRAM_DATA_WIDTH/8)-1 downto 0);
        BRAM_rst : out std_logic;
        BRAM_clk : out std_logic;

        addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);
        data : out std_logic_vector(OUT_DATA_WIDTH-1 downto 0);

        en : in std_logic;
        rst : in std_logic; -- RESET is passed to BRAM, will wipe memory!
        clk : in std_logic
    );    
end bram_slice_fetcher;

architecture Behavioral of bram_slice_fetcher is

    signal sub_word_addr : std_logic_vector(ADDR_WIDTH-BRAM_ADDR_WIDTH-1 downto 0);

begin

    -- Because the BRAM outputs on the rising edge, latch the sub_word_addr 
    process(clk)
    begin
        if rising_edge(clk) then
            sub_word_addr <= addr(ADDR_WIDTH-BRAM_ADDR_WIDTH-1 downto 0);
        end if;
    end process;

    process(BRAM_dout, sub_word_addr) begin
        data <= (others => 'X');
        for i in (BRAM_DATA_WIDTH/OUT_DATA_WIDTH)-1 downto 0 loop
            if sub_word_addr = std_logic_vector(to_unsigned(i, ADDR_WIDTH-BRAM_ADDR_WIDTH)) then
                data <= BRAM_dout((i+1)*OUT_DATA_WIDTH-1 downto i*OUT_DATA_WIDTH);
            end if;
        end loop;
    end process;

    BRAM_addr(32-1 downto ADDR_WIDTH) <= (others => '0');
    BRAM_addr(ADDR_WIDTH-BRAM_ADDR_WIDTH-1 downto 0) <= (others => '0');
    BRAM_addr(ADDR_WIDTH-1 downto ADDR_WIDTH-BRAM_ADDR_WIDTH) <= addr(ADDR_WIDTH-1 downto ADDR_WIDTH-BRAM_ADDR_WIDTH);
    BRAM_din <= (others => '0');
    BRAM_en <= en;
    BRAM_we <= (others => '0');
    BRAM_rst <= rst;
    BRAM_clk <= clk;

end Behavioral;

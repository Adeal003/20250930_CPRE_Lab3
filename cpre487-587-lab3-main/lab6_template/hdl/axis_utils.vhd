----------------------------------------------------------------------------------
-- AXI Stream Utilities
--
-- Gregory Ling, 2024
----------------------------------------------------------------------------------

library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity axis_register_slice is
    generic(
        C_DATA_WIDTH : integer := 32;
        C_TID_WIDTH : integer := 1
    );
    port(
        S_AXIS_TREADY : out std_logic;
        S_AXIS_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS_TLAST  : in  std_logic;
        S_AXIS_TID    : in  std_logic_vector(C_TID_WIDTH-1 downto 0);
        S_AXIS_TVALID : in  std_logic;

        M_AXIS_TREADY : in  std_logic;
        M_AXIS_TDATA  : out std_logic_vector(C_DATA_WIDTH-1 downto 0);
        M_AXIS_TLAST  : out std_logic;
        M_AXIS_TID    : out std_logic_vector(C_TID_WIDTH-1 downto 0);
        M_AXIS_TVALID : out std_logic;

        rst : in std_logic;
        clk : in std_logic
    );
end axis_register_slice;

architecture behavioral of axis_register_slice is

    signal s_pipe_tlast : std_logic;
    signal s_pipe_tvalid : std_logic;
    signal s_pipe_tready : std_logic;
    signal s_pipe_tid : std_logic_vector(C_TID_WIDTH-1 downto 0);
    signal s_pipe_data : std_logic_vector(C_DATA_WIDTH-1 downto 0);

begin

    S_AXIS_TREADY <= s_pipe_tready;

    -- The register is capable of accepting data when it has no data or if all the macs are ready to accept what's currently stored
    s_pipe_tready <= not s_pipe_tvalid or M_AXIS_TREADY;

    process(clk)
    begin
        if rising_edge(clk) then
            if (rst = '1') then
                s_pipe_tlast <= '0';
                s_pipe_tvalid <= '0';
                s_pipe_tid <= (others => '0');
                s_pipe_data <= (others => '0');
            elsif (S_AXIS_TVALID = '1' and s_pipe_tready = '1') then
                s_pipe_tlast <= S_AXIS_TLAST;
                s_pipe_tvalid <= S_AXIS_TVALID;
                s_pipe_tid <= S_AXIS_TID;
                s_pipe_data <= S_AXIS_TDATA;
            elsif (s_pipe_tvalid = '1' and M_AXIS_TREADY = '1') then
                s_pipe_tvalid <= '0';
            end if;
        end if;
    end process;

    M_AXIS_TVALID <= s_pipe_tvalid;
    M_AXIS_TDATA <= s_pipe_data;
    M_AXIS_TLAST <= s_pipe_tlast;
    M_AXIS_TID <= s_pipe_tid;

end architecture behavioral;


library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity axis_4_to_1_round_robin_combiner is
    generic(
        C_DATA_WIDTH : integer := 32
    );
    port(
        S_AXIS0_TREADY : out std_logic;
        S_AXIS0_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS0_TLAST  : in  std_logic;
        S_AXIS0_TVALID : in  std_logic;

        S_AXIS1_TREADY : out std_logic;
        S_AXIS1_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS1_TLAST  : in  std_logic;
        S_AXIS1_TVALID : in  std_logic;

        S_AXIS2_TREADY : out std_logic;
        S_AXIS2_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS2_TLAST  : in  std_logic;
        S_AXIS2_TVALID : in  std_logic;

        S_AXIS3_TREADY : out std_logic;
        S_AXIS3_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS3_TLAST  : in  std_logic;
        S_AXIS3_TVALID : in  std_logic;

        M_AXIS_TREADY : in  std_logic;
        M_AXIS_TDATA  : out std_logic_vector(C_DATA_WIDTH-1 downto 0);
        M_AXIS_TLAST  : out std_logic;
        M_AXIS_TID    : out std_logic_vector(1 downto 0);
        M_AXIS_TVALID : out std_logic;

        rst : in std_logic;
        clk : in std_logic
    );
end axis_4_to_1_round_robin_combiner;

architecture behavioral of axis_4_to_1_round_robin_combiner is

    signal s_index : std_logic_vector(1 downto 0);
    signal s_m_axis_tvalid : std_logic;

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if (rst = '1') then
                s_index <= (others => '0');
            elsif (M_AXIS_TREADY = '1' and s_m_axis_tvalid = '1') then
                s_index <= std_logic_vector(unsigned(s_index) + 1);
            end if;
        end if;
    end process;

    with s_index select M_AXIS_TDATA <= 
        S_AXIS0_TDATA when "00",
        S_AXIS1_TDATA when "01",
        S_AXIS2_TDATA when "10",
        S_AXIS3_TDATA when others;

    with s_index select s_m_axis_tvalid <= 
        S_AXIS0_TVALID when "00",
        S_AXIS1_TVALID when "01",
        S_AXIS2_TVALID when "10",
        S_AXIS3_TVALID when others;
    M_AXIS_TVALID <= s_m_axis_tvalid;

    with s_index select M_AXIS_TLAST <= 
        S_AXIS0_TLAST when "00",
        S_AXIS1_TLAST when "01",
        S_AXIS2_TLAST when "10",
        S_AXIS3_TLAST when others;

    S_AXIS0_TREADY <= '1' when s_index = "00" and M_AXIS_TREADY = '1' else '0';
    S_AXIS1_TREADY <= '1' when s_index = "01" and M_AXIS_TREADY = '1' else '0';
    S_AXIS2_TREADY <= '1' when s_index = "10" and M_AXIS_TREADY = '1' else '0';
    S_AXIS3_TREADY <= '1' when s_index = "11" and M_AXIS_TREADY = '1' else '0';
    
    M_AXIS_TID <= s_index;

end architecture behavioral;


library work;
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity axis_buffered_4_to_1_round_robin_combiner is
    generic(
        C_DATA_WIDTH : integer := 32
    );
    port(
        S_AXIS0_TREADY : out std_logic;
        S_AXIS0_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS0_TLAST  : in  std_logic;
        S_AXIS0_TVALID : in  std_logic;

        S_AXIS1_TREADY : out std_logic;
        S_AXIS1_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS1_TLAST  : in  std_logic;
        S_AXIS1_TVALID : in  std_logic;

        S_AXIS2_TREADY : out std_logic;
        S_AXIS2_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS2_TLAST  : in  std_logic;
        S_AXIS2_TVALID : in  std_logic;

        S_AXIS3_TREADY : out std_logic;
        S_AXIS3_TDATA  : in  std_logic_vector(C_DATA_WIDTH-1 downto 0);
        S_AXIS3_TLAST  : in  std_logic;
        S_AXIS3_TVALID : in  std_logic;

        M_AXIS_TREADY : in  std_logic;
        M_AXIS_TDATA  : out std_logic_vector(C_DATA_WIDTH-1 downto 0);
        M_AXIS_TLAST  : out std_logic;
        M_AXIS_TID    : out std_logic_vector(1 downto 0);
        M_AXIS_TVALID : out std_logic;

        rst : in std_logic;
        clk : in std_logic
    );
end axis_buffered_4_to_1_round_robin_combiner;

architecture behavioral of axis_buffered_4_to_1_round_robin_combiner is

signal s_reg_axis0_tready : std_logic;
signal s_reg_axis0_tdata : std_logic_vector(C_DATA_WIDTH-1 downto 0);
signal s_reg_axis0_tlast : std_logic;
signal s_reg_axis0_tvalid : std_logic;

signal s_reg_axis1_tready : std_logic;
signal s_reg_axis1_tdata : std_logic_vector(C_DATA_WIDTH-1 downto 0);
signal s_reg_axis1_tlast : std_logic;
signal s_reg_axis1_tvalid : std_logic;

signal s_reg_axis2_tready : std_logic;
signal s_reg_axis2_tdata : std_logic_vector(C_DATA_WIDTH-1 downto 0);
signal s_reg_axis2_tlast : std_logic;
signal s_reg_axis2_tvalid : std_logic;

signal s_reg_axis3_tready : std_logic;
signal s_reg_axis3_tdata : std_logic_vector(C_DATA_WIDTH-1 downto 0);
signal s_reg_axis3_tlast : std_logic;
signal s_reg_axis3_tvalid : std_logic;

begin

g_register0: entity work.axis_register_slice
    generic map(
        C_DATA_WIDTH => C_DATA_WIDTH,
        C_TID_WIDTH => 1
    )
    port map(
        S_AXIS_TREADY => S_AXIS0_TREADY, 
        S_AXIS_TDATA => S_AXIS0_TDATA, 
        S_AXIS_TLAST => S_AXIS0_TLAST, 
        S_AXIS_TID => (others => '0'), 
        S_AXIS_TVALID => S_AXIS0_TVALID, 

        M_AXIS_TREADY => s_reg_axis0_tready, 
        M_AXIS_TDATA => s_reg_axis0_tdata, 
        M_AXIS_TLAST => s_reg_axis0_tlast, 
        M_AXIS_TID => open, 
        M_AXIS_TVALID => s_reg_axis0_tvalid,
        
        rst => rst,
        clk => clk
    );

g_register1: entity work.axis_register_slice
    generic map(
        C_DATA_WIDTH => C_DATA_WIDTH,
        C_TID_WIDTH => 1
    )
    port map(
        S_AXIS_TREADY => S_AXIS1_TREADY, 
        S_AXIS_TDATA => S_AXIS1_TDATA, 
        S_AXIS_TLAST => S_AXIS1_TLAST, 
        S_AXIS_TID => (others => '0'), 
        S_AXIS_TVALID => S_AXIS1_TVALID, 

        M_AXIS_TREADY => s_reg_axis1_tready, 
        M_AXIS_TDATA => s_reg_axis1_tdata, 
        M_AXIS_TLAST => s_reg_axis1_tlast, 
        M_AXIS_TID => open, 
        M_AXIS_TVALID => s_reg_axis1_tvalid,

        rst => rst,
        clk => clk
    );

g_register2: entity work.axis_register_slice
    generic map(
        C_DATA_WIDTH => C_DATA_WIDTH,
        C_TID_WIDTH => 1
    )
    port map(
        S_AXIS_TREADY => S_AXIS2_TREADY, 
        S_AXIS_TDATA => S_AXIS2_TDATA, 
        S_AXIS_TLAST => S_AXIS2_TLAST, 
        S_AXIS_TID => (others => '0'), 
        S_AXIS_TVALID => S_AXIS2_TVALID, 

        M_AXIS_TREADY => s_reg_axis2_tready, 
        M_AXIS_TDATA => s_reg_axis2_tdata, 
        M_AXIS_TLAST => s_reg_axis2_tlast, 
        M_AXIS_TID => open, 
        M_AXIS_TVALID => s_reg_axis2_tvalid,
        
        rst => rst,
        clk => clk
    );

g_register3: entity work.axis_register_slice
    generic map(
        C_DATA_WIDTH => C_DATA_WIDTH,
        C_TID_WIDTH => 1
    )
    port map(
        S_AXIS_TREADY => S_AXIS3_TREADY, 
        S_AXIS_TDATA => S_AXIS3_TDATA, 
        S_AXIS_TLAST => S_AXIS3_TLAST, 
        S_AXIS_TID => (others => '0'), 
        S_AXIS_TVALID => S_AXIS3_TVALID, 

        M_AXIS_TREADY => s_reg_axis3_tready, 
        M_AXIS_TDATA => s_reg_axis3_tdata, 
        M_AXIS_TLAST => s_reg_axis3_tlast, 
        M_AXIS_TID => open, 
        M_AXIS_TVALID => s_reg_axis3_tvalid,
        
        rst => rst,
        clk => clk
    );

g_axis_4_to_1_round_robin_combiner: entity work.axis_4_to_1_round_robin_combiner
    generic map(
        C_DATA_WIDTH => C_DATA_WIDTH
    )
    port map(
        S_AXIS0_TREADY => s_reg_axis0_tready,
        S_AXIS0_TDATA => s_reg_axis0_tdata,
        S_AXIS0_TLAST => s_reg_axis0_tlast,
        S_AXIS0_TVALID => s_reg_axis0_tvalid,

        S_AXIS1_TREADY => s_reg_axis1_tready,
        S_AXIS1_TDATA => s_reg_axis1_tdata,
        S_AXIS1_TLAST => s_reg_axis1_tlast,
        S_AXIS1_TVALID => s_reg_axis1_tvalid,

        S_AXIS2_TREADY => s_reg_axis2_tready,
        S_AXIS2_TDATA => s_reg_axis2_tdata,
        S_AXIS2_TLAST => s_reg_axis2_tlast,
        S_AXIS2_TVALID => s_reg_axis2_tvalid,

        S_AXIS3_TREADY => s_reg_axis3_tready,
        S_AXIS3_TDATA => s_reg_axis3_tdata,
        S_AXIS3_TLAST => s_reg_axis3_tlast,
        S_AXIS3_TVALID => s_reg_axis3_tvalid,

        M_AXIS_TREADY => M_AXIS_TREADY,
        M_AXIS_TDATA => M_AXIS_TDATA,
        M_AXIS_TLAST => M_AXIS_TLAST,
        M_AXIS_TID => M_AXIS_TID,
        M_AXIS_TVALID => M_AXIS_TVALID,

        clk => clk,
        rst => rst
    );

end architecture behavioral;

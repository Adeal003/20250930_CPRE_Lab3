relaunch_sim

# Clock and reset
add_force {/piped_mac/ACLK} -radix hex {0 0ns} {1 5000ps} -repeat_every 10000ps
add_force {/piped_mac/ARESETN} -radix hex {0 0ns}
run 30ns
add_force {/piped_mac/ARESETN} -radix hex {1 0ns}
# No back pressure
add_force {/piped_mac/MO_AXIS_TREADY} -radix hex {1 0ns}


# 1. Basic Single Operation
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {FFFF000022220000 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50ns
# Expected:
#   0xDDDE_0000

# 1. Basic Single Operation (Underflow)
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {8111000022220000 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50ns
# Expected:
#   0x8000_0000 [Underflow]

# 1. Basic Single Operation (Overflow)
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {1111111122222222 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50ns
# Expected:
#   0x7FFF_FFFF [Overflow]


# 2. Basic Double Operation
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {0 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {2 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0011111100222222 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0010101000202020 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {0 0ns}
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50ns
# Expected:
#   0x044A_90CE


# 3. Basic Single Operation back-to-back
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {3 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000AAAA0000BBBB 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {4 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000444400005555 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50ns
# Expected:
#   0x0000_7D26
#   0x0000_16C1 ****


# 4. Inital Load Single Operation ********
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {0 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {5 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000000000001111 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {1 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TID} -radix hex {6 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000AAAA0000BBBB 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50ns
# Expected:
#   0x0000_8E37 *****


# 5. Basic Multi Operation
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {0 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {7 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000AAAA0000BBBB 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {8 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000222200003333 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 30ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {9 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0000444400001101 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 50 ns
# Expected:
#   0x0000_8883


# Apply Back Pressure
add_force {/piped_mac/MO_AXIS_TREADY} -radix hex {1 0ns}
run 10ns

# 6. Basic Single Operation
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TLAST} -radix hex {1 0ns}
add_force {/piped_mac/SD_AXIS_TID} -radix hex {A 0ns}
add_force {/piped_mac/SD_AXIS_TDATA} -radix hex {0044455500333111 0ns}
add_force {/piped_mac/SD_AXIS_TUSER} -radix hex {0 0ns}
run 10ns
add_force {/piped_mac/SD_AXIS_TVALID} -radix hex {0 0ns}
run 30ns
# Aleaviate Back Pressure
add_force {/piped_mac/MO_AXIS_TREADY} -radix hex {0 0ns}
run 50ns
# Expected:
#   0x0DA6_E5BC


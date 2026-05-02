0x80052b70 - start of shit
...
0x80052bdc - check for $v0

within this block the bios loads something from 0x800dedf0 (at 0x80052b78,
double check that load address is constant) and does some random shifts on it.
across calls it acts like a counter, but for some reason on pchan it loops back
to 0 when it shouln't. fixing this should get audio working.

the memory at 0x800dedf0 is largely what it should be. but the emulator
stores half of $v0 (sh instruction) at 0x80052f58, and $v0 should be 4 (and
increasing), but it is 0.

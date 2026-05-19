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

at 0x800524dc, register $t0 is the incrementing counter from which the state of
$v0 is derived. again, it resets to 0 instead of going to 4.

there seems to be some kind of memory corruption at address `0x800dea6e` as
instruction
```asm
sh $t0, $t2, 0x0006 ; @ 0x800531b4
```
writes `0x0` instead of `0xe347` (`$t0` is indeed 0). in fact all subsequent
writes are 0.

FINALLY: $t0 is loaded from `0x1f801c1c`, io port for SPU voice 1 current adsr
volume. which i dont implement :(.

# Jump Delay Slot Crash

exec history:

```
0x8004f3d0 
0x800312a4 <- move $a1, $s0 in delay slot
0x8004f434
0x8004fac4 <- crash here
```

# Bad Gpu dma madr

write to madr happens at 0x80050760. address is actually perfectly fine! problem
is pchan ram is empty at 0x801b6c24, but there should be data there. data is
supposed to be written at that address in ram via vram to ram dma (i think).

the flow should be:
- clear area (works fine)
- render a bunch of polygons in a circle to create those balls (does not happen?)
- blit the rendered balls to ram (works fine, but vram is zero)

the draw calls are flushed AFTER the blit happens. when reading or writing to vram,
we need to flush draw commands.

fixed.

# Sio0 kernel ISR never sending 0x42

the kernel successfuly reaches 0x000045e4, at which point loading 0x42 into
register $t2 should be guaranteed. next it should write that to sio0 tx. 
- [x] $t2 is 0x42 in the next block, correct

at 0x000045f0 (block 0x000045e4) the kernel checks for bit I_STAT.07 (controller
irq). it seems that the bit never gets set, even though the irq *does* trigger
in the logs?

the write of 0x42 happens at 0x0000462c, after a loop that waits for I_STAT.07
to be set. There seems to be a timing issue in pchan. before the loop an ack
to I_STAT.07 happens. this note from psx-spx seems relevant:
```
After sending a byte, the Kernel waits 100 cycles or so, and does THEN
acknowledge any old IRQ7, and does then wait for the new IRQ7. Due to that
bizarre coding, emulators can't trigger IRQ7 immediately within 0 cycles after
sending the byte.
```



# P-chan 🐷🎀
*Pーちゃん* 

WIP high performance PlayStation 1 emulator

<div align="center">
  <img src="./recording.gif" width=50%></img>
</div>

## Build

> By default P-Chan compiles with opt-level=1. If you want faster compile times, pass `-C opt-level=0` to `RUSTFLAGS`

For the tui debugger:

```sh
cargo run -p pchan-dbg-v2
# or to run an executable
cargo run -p pchan-dbg-v2 -- game.exe
```

## Status

All games will have some graphical glitches because the gpu isn't done yet.
Games I'm testing:

- MK2: **playable**
- Crash Bandicoot: **playable**, some graphical glitches, maybe timing issues
- Ridge Racer: **intro** (cdrom issue)
- Puzzle Bobble 2: **playable**, very broken graphically 
- FF7: **no boot**
- Silent Hill: **no boot**

Progress:
- [x] memory
  - [x] fastmem (bios and ram)
  - [x] io interface
- [x] dynarec (dynasm-rs based, 95% completed, very few rare instructions left)
  - [x] removed cranelift (way too slow)
  - [x] aarch64
  - [ ] x86_64 (later)
  - [ ] risc-v (later)
- [ ] gpu
  - [x] guest emulation
    - [x] basic functionality (vram, dma, gpu commands, etc)
    - [x] timing
    - [x] draw calls
  - [ ] `WIP` hardware (wgpu) renderer
    - [x] polygons (tris and quads)
    - [x] rects (sprites)
    - [ ] lines
  - [ ] software renderer (not started)
- [ ] gte
  - [ ] gte commands implementation: 55% done
  - [ ] gte correctness
  - [ ] load delays
  - [ ] fifo mirror registers and move-on-write 
- [ ] cdrom
  - [x] basic registers
  - [x] fifo
  - [ ] commands
  - [ ] cd-xa (music)
- [ ] spu
  - [x] basic spu & gauss interpolation
  - [x] audio thread 
  - [x] volume
  - [x] adsr envelope
  - [x] reverb
  - [ ] noise generator
- [ ] sio
  - [x] input
    - [x] digital joypad
    - [ ] analog joypad
  - [ ] memcards

## Milestones

- [x] reset vector
- [x] tty output
- [x] psxtest cpu
- [x] shell
- [x] in-game
- [ ] good compatibility

### Time frame

about 10 years 

## Performance

P-chan can render a frame of Crash Bandicoot (in-game, level 1) in roughly
3ms on an m2 macbook air, which is still roughly 5x slower than Duckstation.
The low hanging fruits for optimization currently are general guest emulation
(mostly the spu) and the hardware rasterizer memory usage (per primitive data is
duplicated per vertex). I also expect a pretty big improvement once I implement
JIT block hot linking. The main focus now however is on features and correctness,
and I'm not too upset by current performance.

A goal of this project is to make a very accurate and fast dynarec, such that an
interpreter is not needed. This might be impossible and/or it might make P-chan
quite cycle-inaccurate. The reason behind this decision is that, I really cannot
be assed to code an interpreter. No interpreter makes debugging difficult, as
such, much <3 to the devs of PCSX-Redux, which I am using for cross comparison.

## Code Policy

P-Chan is free software licensed under the GPL-3 license, and it will always
remain that way.

**No llm generated code** exists in this repo. Never glorify
incompetence, and never rent convenience from corporations.

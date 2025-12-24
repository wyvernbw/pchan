# P-chan
*Pーちゃん* 🐷🎀

WIP high performance PlayStation 1 emulator

## Status

- [x] memory and base of memory mapped IO
- [x] dynarec (95% completed, very few rare instructions left)
  - [x] aarch64
  - [ ] x86_64 (later)
  - [ ] risc-v (later)
- [ ] gpu
- [ ] wgpu renderer
- [ ] cdrom
- [ ] spu

## Milestones

- [x] reset vector
- [x] tty output
- [ ] psxtest cpu
- [ ] shell
- [ ] in-game

## Performance

So far performance is "promising". P-chan can emulate the PSX cpu at more
than 10 times its speed (350mhz vs orginal 33mhz) on my macbook air. Note that
the longer the emulator runs for, the faster it gets as more code is cached and
on average the speed increases (up to a point obviously). Since the emulator
crashes almost instantly (due to `todo!`s) there isn't much time for it to
"accelerate". I expect it to reach up to 1Ghz once it is complete. Even then, im
not sure if these numbers are at all impressive, its quite possible its slower
than other dynarec emulators (exact numbers are not available because nobody
measures speed in frequency, since its kind of a pointless metric, in case you
couldn't tell from this entire paragraph).

The implementation is missing a very
large chunk of things that will inevitably have an effect on performance when
added.

A goal of this project is to make a very accurate and fast dynarec, such that
an interpreter is not needed. This might be impossible and/or it might make
P-chan quite cycle-inaccurate. So far however, (barring the seemingly endless
pile of bugs) the dynarec is both fast enough in the uncached scenario where
an interpreter would be needed, and its accurate enough. The reason behind this
decision is that, I really cannot be assed to code an interpreter.









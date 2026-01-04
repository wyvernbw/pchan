# P-chan 🐷🎀
*Pーちゃん* 

WIP high performance PlayStation 1 emulator

## Build

to build the entire workspace:

```
cargo build -w
```

to run the debugger (i highly recommend release mode)

```
cargo run -p pchan-debug-tui --release
```

## Status

- [x] memory and base of memory mapped IO
- [x] dynarec (dynasm-rs based, 95% completed, very few rare instructions left)
  - [x] removed cranelift (way too slow)
  - [x] aarch64
  - [ ] x86_64 (later)
  - [ ] risc-v (later)
- [ ] gpu
- [ ] wgpu renderer
- [ ] cdrom
- [ ] spu

NOTE: `pchan-cranelift-fronted` is deprecated.

## Milestones

- [x] reset vector
- [x] tty output
- [ ] psxtest cpu
- [ ] shell
- [ ] in-game

### Dynarec

After I make the emulator work in 80% of cases with `dynarec-v2`, i want
to implement a `dynarec-v3` that will use the old one as its base and is
as accurate as BeetlePSX's lightrec (when it's not falling back to the
interpreter) or at least as accurate as DuckStation. i dont want to compromise
on performance.

### Time frame

about 10 years 

## Performance

So far performance is "promising". P-chan can emulate the PSX cpu at more than
10 times its speed (350mhz vs orginal 33mhz) on my macbook air. Note that the
longer the emulator runs for, the faster it gets as more code is cached and
on average the speed increases (up to a point obviously). Since the emulator
crashes almost instantly (due to `todo!`s) there isn't much time for it to
"accelerate". I expect it to reach up to 1Ghz once it is complete. Even then, im
not sure if these numbers are at all impressive, its quite possible its slower
than other dynarec emulators (exact numbers are not available because nobody
measures speed in frequency, since its kind of a pointless metric, in case you
couldn't tell from this entire paragraph).

Switching from cranelift to dynasm-rs made compilation faster by about 2 orders
of magnitude (a lot of the overhead was cranelift but a big part was also
cranelift-isms in my code, including full cfg builidng as opposed to compiling
simple blocks).

The implementation is missing a very
large chunk of things that will inevitably have an effect on performance when
added.

A goal of this project is to make a very accurate and fast dynarec, such that
an interpreter is not needed. This might be impossible and/or it might make
P-chan quite cycle-inaccurate. So far however, (barring the seemingly endless
pile of bugs) the dynarec is both fast enough in the uncached scenario where
an interpreter would be needed, and its accurate enough. The reason behind this
decision is that, I really cannot be assed to code an interpreter.

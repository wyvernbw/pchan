# P-chan 🐷🎀
*Pーちゃん* 

WIP high performance PlayStation 1 emulator

## Status

The current emulator includes the dynarec cpu and hardware rasterizer. Though
not complete yet, they are enough to render the bios splash screen accurately.

- [x] memory and base of memory mapped IO
- [x] dynarec (dynasm-rs based, 95% completed, very few rare instructions left)
  - [x] removed cranelift (way too slow)
  - [x] aarch64
  - [ ] x86_64 (later)
  - [ ] risc-v (later)
- [x] gpu 
- [x] wgpu renderer
- [ ] cdrom
- [ ] spu << we are here
- [ ] input

## Build

to build the entire workspace:

```
cargo build --workspace
```

to run the fully-featured tui debugger (i highly recommend release mode)

```
cargo run -p pchan-dbg --release
```

or for the egui debugger:

```
cargo run -p pchan-dbg-egui
```

## Milestones

- [x] reset vector
- [x] tty output
- [ ] psxtest cpu
- [x] shell
- [ ] in-game

### Dynarec

After I make the emulator work in 80% of cases with `dynarec-v2`, i ideally
want to implement a `dynarec-v3` that will use the old one as its base and
is as accurate as BeetlePSX's lightrec (when it's not falling back to the
interpreter) or at least as accurate as DuckStation. i dont want to compromise
on performance.

### Time frame

about 10 years 

## Performance

The dynarec cpu can run at around 30 times faster than the psx cpu on my macbook
air m2. The slow part is the IO, which i haven't optimized yet, but the emulator
remains fast enough that it can easily keep 60fps when compiled *in debug mode*.
Sampling profilers show that around 70% of the time spent is just sleeping, so I
would say it is pretty efficient so far.

The implementation is missing a very large chunk of things that will inevitably
have an effect on performance when added.

A goal of this project is to make a very accurate and fast dynarec, such that
an interpreter is not needed. This might be impossible and/or it might make
P-chan quite cycle-inaccurate. So far however, (barring the seemingly endless
pile of bugs) the dynarec is both fast enough in the uncached scenario where
an interpreter would be needed, and its accurate enough. The reason behind this
decision is that, I really cannot be assed to code an interpreter.

## On AI Code

**No llm generated code** exists in **any** of the `pchan-*` crates (so all
crates in this repo). I do not condone the creation of machine generated slop.
Such workflows might work for uninspired react copy and paste dashboards,
but emulators require a holistic knowledge of hardware and implementation.
Some dependencies might include llm generated code (trust me I wish I could
go full schizo and remove them, but that isn't really practical sadly). Such
dependencies are usually limited to non core crates (so things outside of
`pchan-emu`, `pchan-gpu`, `pchan-audio` etc.).

As such, any hallucinated code came straight from my human, tired brain :)

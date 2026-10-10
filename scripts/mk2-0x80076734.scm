(require-builtin "pchan/emu")
(require "pchan/main.scm")


(begin
  (await (frames 40))
  (emu.cpu.gpr.$t0))

(require "pchan/emu.scm")
(require-builtin "pchan/emu")

(begin
  (block-on (open-content (env-var "MK2")))
  (set-speed 'unlimited)
  (set-volume 0.0)
  (block-on (frames 4825))
  (define crash (add-breakpoint #x800766b4 'x))
  (displayln "added breakpoint" crash)
  (block-on (breakpoint crash))
  (displayln "$a1=" (emu.cpu.gpr.$a1))
  (displayln "$a3=" (emu.cpu.gpr.$a3)))


(require "pchan/emu.scm")
(require-builtin "pchan/emu")

(begin
  (set-speed 'unlimited)
  ; (set-volume 0.0)
  (block-on (open-content))
  (block-on (frames 4825))
  (define crash (add-breakpoint #x80076734 'x))
  (displayln "added breakpoint" crash)
  (block-on (breakpoint crash))
  (displayln "$a1=" (emu.cpu.gpr.$a1))
  (displayln "$a3=" (emu.cpu.gpr.$a3)))


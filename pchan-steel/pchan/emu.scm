(provide
  run pause hard-reset frames frame block-on
  add-breakpoint del-breakpoint switch-breakpoint breakpoint
  set-volume set-speed open-content
  emu.readu32 emu.cpu.gpr
  u32
  emu.cpu.gpr.$zero emu.cpu.gpr.$at emu.cpu.gpr.$v0 emu.cpu.gpr.$v1
  emu.cpu.gpr.$a0 emu.cpu.gpr.$a1 emu.cpu.gpr.$a2 emu.cpu.gpr.$a3
  emu.cpu.gpr.$t0 emu.cpu.gpr.$t1 emu.cpu.gpr.$t2 emu.cpu.gpr.$t3
  emu.cpu.gpr.$t4 emu.cpu.gpr.$t5 emu.cpu.gpr.$t6 emu.cpu.gpr.$t7
  emu.cpu.gpr.$s0 emu.cpu.gpr.$s1 emu.cpu.gpr.$s2 emu.cpu.gpr.$s3
  emu.cpu.gpr.$s4 emu.cpu.gpr.$s5 emu.cpu.gpr.$s6 emu.cpu.gpr.$s7
  emu.cpu.gpr.$t8 emu.cpu.gpr.$t9 emu.cpu.gpr.$k0 emu.cpu.gpr.$k1
  emu.cpu.gpr.$gp emu.cpu.gpr.$sp emu.cpu.gpr.$fp emu.cpu.gpr.$ra)

(define u32 void)

(define (run) void)
(define (pause) void)
(define (hard-reset) void)
(define (frames n) void)
(define (frame) void)
(define (add-breakpoint address kind) void)
(define (del-breakpoint address) void)
(define (switch-breakpoint address value) void)
(define (breakpoint address) void)
(define (set-volume v) void)
(define (set-speed v) void)
(define (open-content) void)
(define (emu.readu32 address) void)
(define (emu.cpu.gpr) void)
(define (block-on) void)

(define (emu.cpu.gpr.$zero) void)
(define (emu.cpu.gpr.$at) void)
(define (emu.cpu.gpr.$v0) void)
(define (emu.cpu.gpr.$v1) void)
(define (emu.cpu.gpr.$a0) void)
(define (emu.cpu.gpr.$a1) void)
(define (emu.cpu.gpr.$a2) void)
(define (emu.cpu.gpr.$a3) void)
(define (emu.cpu.gpr.$t0) void)
(define (emu.cpu.gpr.$t1) void)
(define (emu.cpu.gpr.$t2) void)
(define (emu.cpu.gpr.$t3) void)
(define (emu.cpu.gpr.$t4) void)
(define (emu.cpu.gpr.$t5) void)
(define (emu.cpu.gpr.$t6) void)
(define (emu.cpu.gpr.$t7) void)
(define (emu.cpu.gpr.$s0) void)
(define (emu.cpu.gpr.$s1) void)
(define (emu.cpu.gpr.$s2) void)
(define (emu.cpu.gpr.$s3) void)
(define (emu.cpu.gpr.$s4) void)
(define (emu.cpu.gpr.$s5) void)
(define (emu.cpu.gpr.$s6) void)
(define (emu.cpu.gpr.$s7) void)
(define (emu.cpu.gpr.$t8) void)
(define (emu.cpu.gpr.$t9) void)
(define (emu.cpu.gpr.$k0) void)
(define (emu.cpu.gpr.$k1) void)
(define (emu.cpu.gpr.$gp) void)
(define (emu.cpu.gpr.$sp) void)
(define (emu.cpu.gpr.$fp) void)
(define (emu.cpu.gpr.$ra) void)

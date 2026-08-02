use pchan_utils::hex;

use crate::Emu;
use crate::cpu::RA;
use crate::cpu::exceptions::Exception;
use crate::cpu::ops::*;
use crate::dynarec_v2::emitters::{DecodedOp, DynarecOp};
use crate::memory::ext;

#[derive(Debug, Clone)]
pub struct Interpreter {
    delay_queue:   [DelaySlot; 2],
    next_op:       (u32, DecodedOp),
    in_delay_slot: bool,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self {
            delay_queue:   [const { DelaySlot::Nop }; 2],
            next_op:       (0x0, DecodedOp::Nop(Nop)),
            in_delay_slot: false,
        }
    }
}

enum MemOpSize {
    Word,
    Halfword,
    Byte,
}

#[derive(derive_more::Debug, Clone, PartialEq, Eq)]
enum DelaySlot {
    Nop,
    Lwl {
        register: u8,
        #[debug("{}", hex(*address))]
        address:  u32,
    },
    Lwr {
        register: u8,
        #[debug("{}", hex(*address))]
        address:  u32,
    },
    SetReg {
        reg:   u8,
        value: u32,
    },
    SetCop {
        cop:   u8,
        idx:   u8,
        value: u32,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum InterpreterResult {
    Exception,
    Hblank,
    Vblank,
    None,
}

impl Interpreter {
    fn debugger_exec(#[allow(unused_variables)] emu: &mut Emu) {
        #[cfg(feature = "debugger-ext")]
        {
            use crate::debug::BreakpointKind;

            emu.dbg.break_on(emu.cpu.pc, BreakpointKind::EXECUTE);
        }
    }

    fn run_delay_slots(&mut self, emu: &mut Emu) {
        emu.run_op_delay_slot(std::mem::replace(&mut self.delay_queue[0], DelaySlot::Nop));
        self.delay_queue.swap(0, 1);
        debug_assert!(self.delay_queue[1] == DelaySlot::Nop);
    }

    pub fn run_instruction(&mut self, emu: &mut Emu) -> (InterpreterResult, u32, DecodedOp) {
        if !emu.cpu.pc.is_multiple_of(0x4) {
            emu.raise_exception(Exception::AdEl);
            emu.run_io();
            return (
                InterpreterResult::Exception,
                emu.cpu.pc,
                DecodedOp::Nop(Nop),
            );
        }

        let in_delay_slot = self.in_delay_slot;
        if in_delay_slot {
            emu.cpu.cop0.set_bd(true);
        }
        let op = self.next_op;
        self.next_op = (
            emu.cpu.pc,
            DecodedOp::new(
                emu.fastmem_read::<OpCode>(emu.cpu.pc)
                    .expect("unhandled instruction fetch"),
            ),
        );
        emu.cpu.pc = emu.cpu.pc.wrapping_add(0x4);
        tracing::trace!(pc=%hex(op.0), op = %op.1);

        self.run_delay_slots(emu);
        let delay_slot = emu.run_op(self, op.0, op.1);
        let d_clock = op.1.cycles() as u64;
        emu.cpu.d_clock += d_clock as u32;
        emu.run_io();
        emu.cpu.d_clock = 0;
        emu.cpu.drain_jump_queue();
        if in_delay_slot {
            self.in_delay_slot = false;
        }
        if let Some(op) = delay_slot {
            debug_assert!(self.delay_queue[1] == DelaySlot::Nop);
            self.delay_queue[1] = op;
        };

        Self::debugger_exec(emu);
        #[cfg(feature = "debugger-ext")]
        if emu.dbg.stopped_on.is_some() {
            return (InterpreterResult::Exception, op.0, op.1);
        }

        if emu.gpu.vblank_signal {
            return (InterpreterResult::Vblank, op.0, op.1);
        }

        #[cfg(test)]
        if matches!(self.next_op.1, DecodedOp::HaltBlock(_)) {
            return (InterpreterResult::Exception, op.0, op.1);
        }

        (InterpreterResult::Exception, op.0, op.1)
        // InterpreterResult::None
    }
}

impl Emu {
    pub(super) fn set_reg(&mut self, idx: u8, value: u32) {
        self.cpu.gpr[idx as usize] = value;
    }
    pub(super) fn get_reg(&self, idx: u8) -> u32 {
        self.cpu.gpr[idx as usize]
    }
    pub(super) fn set_cop(&mut self, cop: u8, idx: u8, value: u32) {
        let idx = idx as usize;
        match cop {
            0 => {
                self.cpu.cop0.reg[idx] = value;
            }
            2 => {
                self.cpu.cop2.reg[idx] = value;
            }
            _ => panic!("invalid cop: {cop}"),
        }
    }
    pub(super) fn get_cop(&self, cop: u8, idx: u8) -> u32 {
        let idx = idx as usize;
        match cop {
            0 => self.cpu.cop0.reg[idx],
            1 => self.cpu.cop1.reg[idx],
            2 => self.cpu.cop2.reg[idx],
            _ => panic!("invalid cop: {cop}"),
        }
    }
    fn run_op_delay_slot(&mut self, op: DelaySlot) {
        // tracing::info!(delay_slot = ?op);
        match op {
            DelaySlot::Nop => {}
            DelaySlot::Lwl { register, address } => {
                let overwrite = self.get_reg(register);
                let value = self.read32_unaligned_l(address, overwrite);
                self.set_reg(register, value);
            }
            DelaySlot::Lwr { register, address } => {
                let overwrite = self.get_reg(register);
                let value = self.read32_unaligned_r(address, overwrite);
                self.set_reg(register, value);
            }
            DelaySlot::SetReg { reg, value } => {
                self.set_reg(reg, value);
            }
            DelaySlot::SetCop { cop, idx, value } => {
                self.set_cop(cop, idx, value);
            }
        }
        self.set_reg(0, 0);
    }
    fn run_op(&mut self, interp: &mut Interpreter, op_pc: u32, op: DecodedOp) -> Option<DelaySlot> {
        let res = match op {
            DecodedOp::Nop(_) => None,
            DecodedOp::Illegal(_) => None,
            DecodedOp::Sll(sll) => {
                self.set_reg(sll.rd, self.get_reg(sll.rt) << sll.shamt);
                None
            }
            DecodedOp::Srl(srl) => {
                self.set_reg(srl.rd, self.get_reg(srl.rt) >> srl.shamt);
                None
            }
            DecodedOp::Sra(sra) => {
                let rt = self.get_reg(sra.rt) as i32;
                self.set_reg(sra.rd, (rt >> sra.shamt) as u32);
                None
            }
            DecodedOp::Sllv(sllv) => {
                let rs = self.get_reg(sllv.rs) & 0x1f;
                self.set_reg(sllv.rd, self.get_reg(sllv.rt) << rs);
                None
            }
            DecodedOp::Srlv(srlv) => {
                let rs = self.get_reg(srlv.rs) & 0x1f;
                self.set_reg(srlv.rd, self.get_reg(srlv.rt) >> rs);
                None
            }
            DecodedOp::Srav(srav) => {
                let rs = self.get_reg(srav.rs) & 0x1f;
                let rt = self.get_reg(srav.rt) as i32;
                self.set_reg(srav.rd, (rt >> rs) as u32);
                None
            }
            DecodedOp::Jr(jr) => {
                self.jump(interp, self.get_reg(jr.rs));
                None
            }
            DecodedOp::Jalr(jalr) => {
                self.link_return_in(interp, op_pc, jalr.rd);
                self.jump(interp, self.get_reg(jalr.rs));
                None
            }
            DecodedOp::Syscall(_) => {
                self.handle_syscall(false);
                // interp.in_delay_slot = true;
                if interp.in_delay_slot {
                    todo!("syscall in delay slot!!!");
                }
                None
            }
            DecodedOp::Mfhi(mfhi) => {
                let hi = self.cpu.hilo >> 32;
                self.set_reg(mfhi.rd, hi as u32);
                None
            }
            DecodedOp::Mthi(mthi) => {
                let hi = self.get_reg(mthi.rs);
                self.cpu.hilo &= 0x0000_0000_ffff_ffff;
                self.cpu.hilo |= (hi as u64) << 32;
                None
            }
            DecodedOp::Mflo(mflo) => {
                let lo = self.cpu.hilo & 0xffff_ffff;
                self.set_reg(mflo.rd, lo as u32);
                None
            }
            DecodedOp::Mtlo(mtlo) => {
                let lo = self.get_reg(mtlo.rs);
                self.cpu.hilo &= 0xffff_ffff_0000_0000;
                self.cpu.hilo |= lo as u64;
                None
            }
            DecodedOp::Mult(mult) => {
                let rs = self.get_reg(mult.rs) as i32 as i64;
                let rt = self.get_reg(mult.rt) as i32 as i64;
                self.cpu.hilo = (rs * rt) as u64;
                None
            }
            DecodedOp::Multu(multu) => {
                let rs = self.get_reg(multu.rs) as u64;
                let rt = self.get_reg(multu.rt) as u64;
                self.cpu.hilo = rs * rt;
                None
            }
            DecodedOp::Div(div) => {
                let rs = self.get_reg(div.rs) as i32;
                let rt = self.get_reg(div.rt) as i32;
                let (hi, lo) = match (rs, rt) {
                    (0.., 0) => (rs, -1),
                    (..0, 0) => (rs, 1),
                    (-0x80000000, -1) => (0, -0x80000000),
                    _ => (rs % rt, rs / rt),
                };
                let hi = hi as u32 as u64;
                let lo = lo as u32 as u64;
                self.cpu.hilo = (hi << 32) | lo;
                None
            }
            DecodedOp::Divu(divu) => {
                let rs = self.get_reg(divu.rs);
                let rt = self.get_reg(divu.rt);
                let (hi, lo) = match (rs, rt) {
                    (_, 0) => (rs, u32::MAX),
                    _ => (rs % rt, rs / rt),
                };
                let hi = hi as u64;
                let lo = lo as u64;
                self.cpu.hilo = (hi << 32) | lo;
                None
            }
            DecodedOp::Addu(addu) => {
                let rs = self.get_reg(addu.rs);
                let rt = self.get_reg(addu.rt);
                self.set_reg(addu.rd, rs.wrapping_add(rt));
                None
            }
            DecodedOp::Subu(subu) => {
                let rs = self.get_reg(subu.rs);
                let rt = self.get_reg(subu.rt);
                self.set_reg(subu.rd, rs.wrapping_sub(rt));
                None
            }
            DecodedOp::And(and) => {
                let rs = self.get_reg(and.rs);
                let rt = self.get_reg(and.rt);
                self.set_reg(and.rd, rs & rt);
                None
            }
            DecodedOp::Or(or) => {
                let rs = self.get_reg(or.rs);
                let rt = self.get_reg(or.rt);
                self.set_reg(or.rd, rs | rt);
                None
            }
            DecodedOp::Xor(xor) => {
                let rs = self.get_reg(xor.rs);
                let rt = self.get_reg(xor.rt);
                self.set_reg(xor.rd, rs ^ rt);
                None
            }
            DecodedOp::Bltz(bltz) => {
                if (self.get_reg(bltz.rs) as i32) < 0 {
                    self.branch(interp, op_pc, bltz.imm16);
                }
                None
            }
            DecodedOp::Bgez(bgez) => {
                if (self.get_reg(bgez.rs) as i32) >= 0 {
                    self.branch(interp, op_pc, bgez.imm16);
                }
                None
            }
            DecodedOp::Bltzal(bltzal) => {
                self.link_return(interp, op_pc);
                if (self.get_reg(bltzal.rs) as i32) < 0 {
                    self.branch(interp, op_pc, bltzal.imm16);
                }
                None
            }
            DecodedOp::Bgezal(bgezal) => {
                self.link_return(interp, op_pc);
                if (self.get_reg(bgezal.rs) as i32) >= 0 {
                    self.branch(interp, op_pc, bgezal.imm16);
                }
                None
            }
            DecodedOp::J(j) => {
                self.jump(interp, (self.cpu.pc & 0xf000_0000) + (j.imm26 << 2));
                None
            }
            DecodedOp::Jal(jal) => {
                self.link_return(interp, op_pc);
                self.jump(interp, (self.cpu.pc & 0xf000_0000) + (jal.imm26 << 2));
                None
            }
            DecodedOp::Blez(blez) => {
                if (self.get_reg(blez.rs) as i32) <= 0 {
                    self.branch(interp, op_pc, blez.imm16);
                }
                None
            }
            DecodedOp::Bgtz(bgtz) => {
                if (self.get_reg(bgtz.rs) as i32) > 0 {
                    self.branch(interp, op_pc, bgtz.imm16);
                }
                None
            }
            DecodedOp::Nor(nor) => {
                let rs = self.get_reg(nor.rs);
                let rt = self.get_reg(nor.rt);
                self.set_reg(nor.rd, !(rs | rt));
                None
            }
            DecodedOp::Slt(slt) => {
                let rs = self.get_reg(slt.rs) as i32;
                let rt = self.get_reg(slt.rt) as i32;
                let rd = rs < rt;
                self.set_reg(slt.rd, rd as u32);
                None
            }
            DecodedOp::Sltu(sltu) => {
                let rs = self.get_reg(sltu.rs);
                let rt = self.get_reg(sltu.rt);
                let rd = rs < rt;
                self.set_reg(sltu.rd, rd as u32);
                None
            }
            DecodedOp::Beq(beq) => {
                let rs = self.get_reg(beq.rs);
                let rt = self.get_reg(beq.rt);
                if rs == rt {
                    self.branch(interp, op_pc, beq.imm16);
                }
                None
            }
            DecodedOp::Bne(bne) => {
                let rs = self.get_reg(bne.rs);
                let rt = self.get_reg(bne.rt);
                if rs != rt {
                    self.branch(interp, op_pc, bne.imm16);
                }
                None
            }
            DecodedOp::HaltBlock(_) => None,
            DecodedOp::Sb(sb) => {
                let value = self.get_reg(sb.rt) as u8;
                let address = self.get_reg(sb.rs).wrapping_add_signed(sb.imm16 as i32);
                self.write(address, value);
                None
            }
            DecodedOp::Sh(sh) => {
                let value = self.get_reg(sh.rt) as u16;
                let address = self.get_reg(sh.rs).wrapping_add_signed(sh.imm16 as i32);
                self.write(address, value);
                None
            }
            DecodedOp::Swl(swl) => {
                let value = self.get_reg(swl.rt);
                let address = self.get_reg(swl.rs).wrapping_add_signed(swl.imm16 as i32);
                self.write32_unaligned_l(address, value);
                None
            }
            DecodedOp::Sw(sw) => {
                let value = self.get_reg(sw.rt);
                let address = self.get_reg(sw.rs).wrapping_add_signed(sw.imm16 as i32);
                self.write(address, value);
                None
            }
            DecodedOp::Swr(swr) => {
                let value = self.get_reg(swr.rt);
                let address = self.get_reg(swr.rs).wrapping_add_signed(swr.imm16 as i32);
                self.write32_unaligned_r(address, value);
                None
            }
            DecodedOp::Lwcn(lwcn) => {
                let address = self.get_reg(lwcn.rs).wrapping_add_signed(lwcn.imm16 as i32);
                let value = self.read::<u32>(address);
                Some(DelaySlot::SetCop {
                    cop: lwcn.cop,
                    idx: lwcn.rt,
                    value,
                })
            }
            DecodedOp::Swcn(_swcn) => {
                // let address = self.get_reg(swcn.rs).wrapping_add_signed(swcn.imm16 as i32);
                // let value = self.get_cop(swcn.cop, swcn.rt);
                // Some(DelaySlot::StoreWord { value, address })
                None
            }
            DecodedOp::Addiu(addiu) => {
                self.set_reg(
                    addiu.rt,
                    self.get_reg(addiu.rs)
                        .wrapping_add_signed(addiu.imm16 as i32),
                );
                None
            }
            DecodedOp::Slti(slti) => {
                let rs = self.get_reg(slti.rs) as i32;
                let rt = rs < slti.imm16 as i32;
                self.set_reg(slti.rt, rt as u32);
                None
            }
            DecodedOp::Sltiu(sltiu) => {
                let rs = self.get_reg(sltiu.rs);
                let rt = rs < ext::sign(sltiu.imm16) as u32;
                self.set_reg(sltiu.rt, rt as u32);
                None
            }
            DecodedOp::Andi(andi) => {
                let rs = self.get_reg(andi.rs);
                self.set_reg(andi.rt, rs & ext::zero(andi.imm16));
                None
            }
            DecodedOp::Ori(ori) => {
                let rs = self.get_reg(ori.rs);
                self.set_reg(ori.rt, rs | ext::zero(ori.imm16));
                None
            }
            DecodedOp::Xori(xori) => {
                let rs = self.get_reg(xori.rs);
                self.set_reg(xori.rt, rs ^ ext::zero(xori.imm16));
                None
            }
            DecodedOp::Lui(lui) => {
                self.set_reg(lui.rt, ext::zero(lui.imm16) << 16);
                None
            }
            DecodedOp::Rfe(_) => {
                self.handle_rfe();
                None
            }
            DecodedOp::Mfcn(mfcn) => {
                let value = self.get_cop(mfcn.cop, mfcn.rd);
                Some(DelaySlot::SetReg {
                    reg: mfcn.rt,
                    value,
                })
            }
            DecodedOp::Cfcn(cfcn) => {
                let value = self.get_cop(cfcn.cop, cfcn.rd + 32);
                Some(DelaySlot::SetReg {
                    reg: cfcn.rt,
                    value,
                })
            }
            DecodedOp::Mtcn(mtcn) => {
                let value = self.get_reg(mtcn.rt);
                Some(DelaySlot::SetCop {
                    cop: mtcn.cop,
                    idx: mtcn.rd,
                    value,
                })
            }
            DecodedOp::Ctcn(ctcn) => {
                let value = self.get_reg(ctcn.rt);
                Some(DelaySlot::SetCop {
                    cop: ctcn.cop,
                    idx: ctcn.rd + 32,
                    value,
                })
            }
            DecodedOp::Lb(lb) => {
                let address = self.get_reg(lb.rs).wrapping_add_signed(lb.imm16 as i32);
                let value = ext::sign(self.read::<i8>(address)) as u32;
                Some(DelaySlot::SetReg { value, reg: lb.rt })
            }
            DecodedOp::Lbu(lbu) => {
                let address = self.get_reg(lbu.rs).wrapping_add_signed(lbu.imm16 as i32);
                let value = ext::zero(self.read::<u8>(address));
                Some(DelaySlot::SetReg { value, reg: lbu.rt })
            }
            DecodedOp::Lh(lh) => {
                let address = self.get_reg(lh.rs).wrapping_add_signed(lh.imm16 as i32);
                let value = ext::sign(self.read::<i16>(address)) as u32;
                Some(DelaySlot::SetReg { value, reg: lh.rt })
            }
            DecodedOp::Lhu(lhu) => {
                let address = self.get_reg(lhu.rs).wrapping_add_signed(lhu.imm16 as i32);
                let value = ext::zero(self.read::<u16>(address));
                Some(DelaySlot::SetReg { value, reg: lhu.rt })
            }
            DecodedOp::Lwr(lwr) => {
                let address = self.get_reg(lwr.rs).wrapping_add_signed(lwr.imm16 as i32);
                Some(DelaySlot::Lwr {
                    register: lwr.rt,
                    address,
                })
            }
            DecodedOp::Lwl(lwl) => {
                let address = self.get_reg(lwl.rs).wrapping_add_signed(lwl.imm16 as i32);
                Some(DelaySlot::Lwl {
                    register: lwl.rt,
                    address,
                })
            }
            DecodedOp::Lw(lw) => {
                let address = self.get_reg(lw.rs).wrapping_add_signed(lw.imm16 as i32);
                let value = self.read::<u32>(address);
                Some(DelaySlot::SetReg { value, reg: lw.rt })
            }
        };
        self.cpu.gpr[0] = 0;
        res
    }

    fn jump(&mut self, interp: &mut Interpreter, addr: u32) {
        self.cpu.pc = addr;
        interp.in_delay_slot = true;
    }

    fn branch(&mut self, interp: &mut Interpreter, base: u32, offset: i16) {
        let dest = base
            .wrapping_add(4)
            .wrapping_add_signed((offset as i32) << 2);
        self.cpu.pc = dest;
        interp.in_delay_slot = true;
    }

    fn link_return(&mut self, interp: &mut Interpreter, op_pc: u32) {
        self.link_return_in(interp, op_pc, RA);
    }

    fn link_return_in(&mut self, _: &mut Interpreter, op_pc: u32, reg: u8) {
        self.set_reg(reg, op_pc + 8);
    }
}

#[cfg(test)]
mod interp_tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::cpu::ops::*;
    use crate::cpu::program;
    use crate::run::Runner;

    #[rstest]
    fn load_delay_01() {
        let mut runner = Runner::new();
        let mut emu: Emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([
                addiu(8, 0, 69),
                sw(8, 0, 0x100),
                nop(),
                lw(9, 0, 0x100),
                nop(),
                nop(),
                OpCode::HALT,
            ]),
        );
        runner.execute(&mut emu);
        assert_eq!(emu.get_reg(9), 69)
    }
    #[rstest]
    fn load_delay_02() {
        let mut runner = Runner::new();
        let mut emu: Emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([
                addiu(8, 0, 69),
                sw(8, 0, 0x100),
                lw(9, 0, 0x100),
                nop(),
                nop(),
                OpCode::HALT,
            ]),
        );
        runner.execute(&mut emu);
        assert_eq!(emu.get_reg(9), 69)
    }
    #[rstest]
    fn load_delay_03() {
        setup_tracing();
        let mut runner = Runner::new();
        let mut emu: Emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([
                addiu(8, 0, 0x100),
                addiu(9, 0, 69),
                sw(8, 0, 0x100),
                sw(9, 0, 0x200),
                nop(),
                addiu(9, 0, 0),
                lw(9, 0, 0x100),
                nop(),
                lw(9, 9, 0x100),
                nop(),
                nop(),
                OpCode::HALT,
            ]),
        );
        runner.execute(&mut emu);
        assert_eq!(emu.read::<u32>(0x200), 69, "expected 69 in memory");
        assert_eq!(emu.get_reg(9), 69, "expected 69 in register")
    }

    #[cfg(test)]
    #[rstest]
    #[case::div_00(div, (8, 10), (9, 2), 0x00000000_00000005)]
    #[case::div_01(div, (0, 0), (8, 3), 0x00000000_00000000)]
    #[case::div_02(div, (8, 2), (9, 0), 0x00000002_ffffffff)]
    #[case::div_03(div, (8, -2i32 as u32), (9, 0), 0xfffffffe_00000001)]
    #[case::div_04(div, (8, -10i32 as u32), (9, 2), 0x00000000_fffffffb)]
    #[case::divu_00(divu, (8, 10), (9, 2), 0x00000000_00000005)]
    #[case::divu_01(divu, (8, 2), (9, 0), 0x00000002_ffffffff)]
    #[case::mult_00(mult, (8, 2), (9, 15), 30)]
    #[case::mult_01(mult, (8, 2), (9, -15i32 as u32), -30i32 as u64)]
    #[case::multu(multu, (8, 2), (9, 15), 30)]
    /// form: {inst} ${reg1}={value1}, ${reg2}={value2} ; assert hilo = {expected}
    pub fn test_mul_div(
        #[case] instr: impl Fn(u8, u8) -> OpCode,
        #[case] rs: (u8, u32),
        #[case] rt: (u8, u32),
        #[case] expected: u64,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
        use crate::cpu::interp::{Interpreter, InterpreterResult};
        use crate::cpu::program;

        use assert_hex::*;
        use pchan_utils::setup_tracing;

        setup_tracing();
        let mut emu = Emu::default();

        emu.cpu.hilo = 0xDEAD_BEEF_DEAD_BEEF;
        emu.cpu.gpr[rs.0 as usize] = rs.1;
        emu.cpu.gpr[rt.0 as usize] = rt.1;
        assert_eq!(emu.cpu.gpr[0], 0);

        emu.write_many(0x0, &program([instr(rs.0, rt.0), OpCode::HALT]));
        let mut interp = Interpreter::default();
        while let (InterpreterResult::None, _, _) = interp.run_instruction(&mut emu) {}

        assert_eq_hex!(emu.cpu.hilo, expected);
        Ok(())
    }

    #[test]
    pub fn test_la() {
        let mut runner = Runner::new();
        let mut emu: Emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([lui(8, 0), addiu(8, 8, 0x0c80), OpCode::HALT]),
        );
        runner.execute(&mut emu);
        assert_eq!(emu.get_reg(8), 0x0c80)
    }
}

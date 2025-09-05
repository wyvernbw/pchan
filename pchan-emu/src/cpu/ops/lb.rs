use crate::cpu::ops::{EmitSummary, Op};
use crate::cranelift_bs::*;
use crate::memory::MapAddress;

use super::{Opcode, PrimaryOp};

pub(crate) struct LB {
    rt: usize,
    rs: usize,
    imm: i16,
}

impl Op for LB {
    fn try_from_opcode(opcode: Opcode) -> Result<Self, Self::TryFromError> {
        let opcode = opcode.as_primary(PrimaryOp::LB)?;
        Ok(LB {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }

    fn emit_ir(&self, mut state: super::EmitParams<'_>) -> Option<EmitSummary> {
        // get pointer to memory passed as argument to the function
        let mem_ptr = state.memory();

        // get cached register if possible, otherwise load it in
        let rs = state.emit_get_register(self.rs);
        let address = rs.as_u32().map().0;

        let rt = state.fn_builder.ins().load(
            types::I8,
            MemFlags::new(),
            mem_ptr,
            address.wrapping_add_signed(self.imm as i32) as i32,
        );
        Some(EmitSummary {
            register_updates: vec![(self.rt, rt)].into_boxed_slice(),
        })
    }

    fn is_block_boundary(&self) -> bool {
        false
    }
}

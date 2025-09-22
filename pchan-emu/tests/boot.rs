#![feature(try_blocks)]
#![allow(unused_variables)]

use std::io::Write;

use inquire::Confirm;
use pchan_emu::{Emu, dynarec::JitSummary, memory::kb};
use pchan_utils::setup_tracing;
use rstest::rstest;

#[rstest]
pub fn boot(setup_tracing: ()) -> color_eyre::Result<()> {
    let mut emu = Emu::default();
    let interactive = Confirm::new("run as interactive shell?")
        .with_default(false)
        .prompt()
        .unwrap();
    emu.load_bios()?;
    emu.jump_to_bios();
    // tracing::info!("exception handler code:");
    // for i in (0xBFC0_0000u32..(0xBFC0_0180 + 10 * 32)).step_by(4) {
    //     let op = emu.mem.read::<u32>(KSEG1Addr(i));
    //     let op = DecodedOp::try_from(pchan_emu::cpu::ops::OpCode(op));
    //     if let Ok(op) = op {
    //         tracing::info!(at = %format!("0x{:08X}", i), %op);
    //     }
    // }
    // return Ok(());
    let result: color_eyre::Result<()> = try {
        loop {
            let summary = emu.step_jit_summarize::<JitSummary>()?;
            if interactive || summary.panicked {
                let prompt = Confirm::new("show dynarec summary?")
                    .with_default(false)
                    .prompt();
                if let Ok(true) = prompt {
                    tracing::info!("{:?}", summary);
                    let prompt = Confirm::new("continue?")
                        .with_default(false)
                        .prompt()
                        .unwrap();
                    if !prompt {
                        return Ok(());
                    }
                }
            }
        }
    };
    if let Err(err) = result {
        tracing::info!(?err);
        tracing::info!(?emu.cpu.pc);
    }

    let memory = &emu.mem.as_ref()[0..kb(2048 + 8192)];
    let mut file = std::fs::File::create("../mem.dump")?;
    file.write_all(memory)?;

    Ok(())
}

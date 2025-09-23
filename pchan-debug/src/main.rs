#![feature(try_blocks)]
#![allow(unused_variables)]

use std::collections::HashSet;

use color_eyre::eyre::OptionExt;
use color_eyre::owo_colors::*;
use inquire::{Confirm, Select};
use pchan_emu::{Emu, dynarec::JitSummary};
use pchan_utils::hex;
use pchan_utils::setup_tracing;
use strum::IntoEnumIterator;

pub fn main() -> color_eyre::Result<()> {
    setup_tracing();

    let mut emu = Emu::default();

    println!("🐗 Running pchan debug tool");

    let interactive = Confirm::new("run as interactive shell?")
        .with_default(false)
        .prompt()
        .unwrap();

    emu.load_bios()?;
    emu.jump_to_bios();

    let mut breakpoints = HashSet::new();
    let mut running = false;

    let result: color_eyre::Result<()> = try {
        'outer: loop {
            let old_pc = emu.cpu.pc;
            let summary = emu.step_jit_summarize::<JitSummary>()?;
            if !interactive {
                continue;
            }
            if breakpoints.contains(&old_pc) {
                tracing::info!(address = %hex(emu.cpu.pc), "🐗 ! breakpoint");
                running = false;
            }
            if running {
                continue;
            }
            let dbg_cmd = debug_menu(&emu, &mut breakpoints, &mut running, Some(summary))?;
            match dbg_cmd {
                DebugMenuCommand::Exit => break 'outer,
                DebugMenuCommand::Step => {}
            }
        }
    };
    if let Err(err) = result {
        tracing::info!(?err);
        tracing::info!(?emu.cpu.pc);
    }

    Ok(())
}

pub enum DebugMenuCommand {
    Exit,
    Step,
}

fn debug_menu(
    emu: &Emu,
    breakpoints: &mut HashSet<u32>,
    running: &mut bool,
    summary: Option<JitSummary>,
) -> Result<DebugMenuCommand, color_eyre::eyre::Error> {
    loop {
        let action = Select::new(
            "menu:",
            StepCommand::iter()
                .filter(|action| !(summary.is_none() && matches!(action, StepCommand::ShowSummary)))
                .collect(),
        )
        .prompt();
        let action = match action {
            Ok(action) => action,
            Err(inquire::InquireError::OperationCanceled) => continue,
            Err(inquire::InquireError::OperationInterrupted) => return Ok(DebugMenuCommand::Exit),
            other => other?,
        };
        match action {
            StepCommand::Step => return Ok(DebugMenuCommand::Step),
            StepCommand::ShowSummary => tracing::info!("{:?}", summary),
            StepCommand::ViewFunction => {
                _ = view_function(emu);
            }
            StepCommand::AddBreakpoint => {
                _ = add_breakpoint(breakpoints);
            }
            StepCommand::Exit => return Ok(DebugMenuCommand::Exit),
            StepCommand::Run => {
                *running = true;
                return Ok(DebugMenuCommand::Step);
            }
        }
    }
}

#[derive(derive_more::Display, strum::EnumIter)]
pub enum StepCommand {
    #[display("step")]
    Step,
    #[display("run")]
    Run,
    #[display("show summary")]
    ShowSummary,
    #[display("view function")]
    ViewFunction,
    #[display("add breakpoint")]
    AddBreakpoint,
    #[display("exit")]
    Exit,
}

fn view_function(emu: &Emu) -> color_eyre::Result<()> {
    #[derive(derive_more::Display)]
    #[display("0x{}", pchan_utils::hex(self.inner))]
    struct DisplayHex<T: Copy> {
        pub inner: T,
    }

    impl<T: Copy> DisplayHex<T> {
        const fn new(inner: T) -> Self {
            Self { inner }
        }
    }

    let functions = emu
        .jit_cache
        .fn_map
        .functions()
        .map(|(address, _)| DisplayHex::new(*address))
        .collect();
    let DisplayHex { inner: select } = Select::new("choose function", functions).prompt()?;
    let function = emu
        .jit_cache
        .fn_map
        .get(select)
        .expect("could not find function that i just got from the map...?");
    tracing::info!("{:#?}", function.func);

    Ok(())
}

fn add_breakpoint(breakpoints: &mut HashSet<u32>) -> color_eyre::Result<()> {
    let breakpoint = inquire::Text::new("type an address")
        .with_placeholder("0xbfc00180")
        .with_help_message("this will not pause exection mid function")
        .prompt()?;
    let address = breakpoint
        .strip_prefix("0x")
        .ok_or_eyre("invalid address")?;
    let address = u32::from_str_radix(address, 16)?;
    breakpoints.insert(address);
    Ok(())
}

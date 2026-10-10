use core::alloc::Allocator;
use core::mem;
use core::num::ParseIntError;
use std::collections::HashMap;

use kanal::{ReceiveError, SendError, Sender};
use pchan_emu::Emu;
use pchan_emu::cpu::{REG_STR, reg_str};
use pchan_emu::debug::Breakpoint;
use pchan_emu::io::UnhandledIO;
use pchan_emu::run::Runner;
use pchan_utils::{AsyncChan, hex};
use steel::gc::Gc;
use steel::rerrs::ErrorKind as SteelErrorKind;
use steel::rvals::{FutureResult, IntoSteelVal, SteelString};
use steel::steel_vm::builtin::BuiltInModule;
use steel::steel_vm::engine::Engine;
use steel::steel_vm::register_fn::RegisterFn;
use steel::{SteelErr, SteelVal};

use crate::primitives::{SteelGprMap, SteelU32, parse_rwx};

mod primitives;

#[derive(Clone)]
pub struct ScriptConn {
    pub chan: AsyncChan<Call>,
}

impl ScriptConn {
    pub fn send_sync(&self, v: Call) -> Result<(), SendError> {
        self.chan.0.as_sync().send(v)
    }
    pub async fn send_async(&self, v: Call) -> Result<(), SendError> {
        self.chan.0.send(v).await
    }
}

pub struct SteelCtx {
    pub engine: Engine,
    pub conn:   ScriptConn,
}

#[derive(Debug, Clone)]
pub enum Call {
    Run,
    Pause,
    HardReset,
    Frame(Sender<isize>),
    WaitFrames(u32, Sender<()>),
    Gpr(Sender<Vec<u32>>),
    GprSingle(u8, Sender<u32>),
    MemReadU32(MemReadCall<u32>),
    AddBreakpoint(Breakpoint),
    DelBreakpoint(u32),
    SwitchBreakpoint(u32, bool, Sender<Result<(), PchanSteelErr>>),
}

#[derive(Debug, Clone)]
pub struct MemReadCall<T: Copy>(u32, Sender<Result<T, UnhandledIO>>);

impl SteelCtx {
    #[must_use]
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let mut module = BuiltInModule::new("pchan/emu");
        let conn = ScriptConn {
            chan: kanal::bounded_async(16),
        };

        engine.register_type::<SteelGprMap>("pchan::SteelGprMap");

        SteelU32::register(&mut module);

        module.register_fn("await", move |value: SteelVal| {
            if let SteelVal::FutureV(f) = value {
                let shared = f.unwrap().into_shared();
                smol::block_on(shared)
            } else {
                Ok(value)
            }
        });

        let c = conn.clone();
        module.register_fn("run", move || c.clone().send_sync(Call::Run).steel());
        let c = conn.clone();
        module.register_fn("pause", move || c.send_sync(Call::Pause).steel());
        let c = conn.clone();
        module.register_fn("hard-reset", move || c.send_sync(Call::HardReset).steel());

        let c = conn.clone();
        module.register_fn("frames", move |n: u32| {
            future(&c, async move |c| {
                let (tx, rx) = kanal::bounded_async(0);
                c.send_async(Call::WaitFrames(n, tx.to_sync()))
                    .await
                    .steel()?;
                rx.recv().await.steel()?;
                Ok(())
            })
        });

        let c = conn.clone();
        module.register_fn("frame", move || -> SteelResult<_> {
            let (tx, rx) = kanal::bounded(0);
            c.send_sync(Call::Frame(tx)).steel()?;
            let res = rx.recv().steel()?;
            Ok(res)
        });

        let c = conn.clone();
        module.register_fn("emu.cpu.gpr", move || {
            let (tx, rx) = kanal::bounded(0);
            c.send_sync(Call::Gpr(tx)).steel()?;
            rx.recv().steel().map(|gpr| {
                gpr.iter()
                    .copied()
                    .enumerate()
                    .map(|(gpr, value)| (REG_STR[gpr], SteelU32(value)))
                    .collect::<HashMap<_, _>>()
            })
        });

        for gpr in 0..32 {
            let fn_name = format!("emu.cpu.gpr.${}", reg_str(gpr));
            let fn_name = Box::leak(Box::new(fn_name));
            let c = conn.clone();
            module.register_fn(fn_name, move || {
                let (tx, rx) = kanal::bounded(0);
                c.send_sync(Call::GprSingle(gpr, tx)).steel()?;
                rx.recv().steel().map(SteelU32)
            });
        }

        let c = conn.clone();
        module.register_fn("emu.readu32", move |address: u32| {
            let (tx, rx) = kanal::bounded(0);
            c.send_sync(Call::MemReadU32(MemReadCall(address, tx)))
                .steel()?;
            rx.recv().steel()?.steel().map(SteelU32)
        });

        let c = conn.clone();
        module.register_fn(
            "add-breakpoint",
            move |address: u32, rwx: SteelVal| -> SteelResult<_> {
                let kind = rwx.symbol_or_else(|| {
                    SteelErr::new(SteelErrorKind::TypeMismatch, "expected symbol".to_owned())
                })?;
                let kind = parse_rwx(kind)?;
                c.send_sync(Call::AddBreakpoint(Breakpoint {
                    address,
                    kind,
                    enabled: true,
                }))
                .steel()?;
                Ok(())
            },
        );

        let c = conn.clone();
        module.register_fn("del-breakpoint", move |address: u32| {
            c.send_sync(Call::DelBreakpoint(address)).steel()
        });

        let c = conn.clone();
        module.register_fn(
            "switch-breakpoint",
            move |address: u32, value: SteelVal| -> SteelResult<_> {
                let value = match value {
                    SteelVal::SymbolV(sym) => match sym.as_str() {
                        "on" => Ok(true),
                        "off" => Ok(false),
                        _ => Err(SteelErr::new(
                            SteelErrorKind::Parse,
                            "expected either 'on' or 'off'.".to_owned(),
                        )),
                    },
                    SteelVal::BoolV(value) => Ok(value),
                    _ => Err(SteelErr::new(
                        SteelErrorKind::TypeMismatch,
                        "expected bool or 'on/'off.".to_owned(),
                    )),
                };
                let value = value?;
                let (tx, rx) = kanal::bounded(0);
                c.send_sync(Call::SwitchBreakpoint(address, value, tx))
                    .steel()?;
                rx.recv().steel()?.steel()
            },
        );

        engine.register_module(module);
        engine.run(r#"(require-builtin "pchan/emu")"#).unwrap();

        Self { engine, conn }
    }

    #[must_use]
    pub fn rx(&self) -> ScriptConn {
        self.conn.clone()
    }

    #[cfg(feature = "repl")]
    pub fn repl(self) -> std::io::Result<()> {
        steel_repl::run_repl(self.engine)
    }
}

fn parse_hex_word(str: &str) -> Result<u32, ParseIntError> {
    if str == "0x" {
        return Ok(0);
    }
    let str = str.trim_prefix("0x");
    if let Some((a, b)) = str.split_once('_') {
        let a = parse_hex_word(a)?;
        let b = parse_hex_word(b)?;
        Ok(a << 16 | b)
    } else {
        u32::from_str_radix(str, 16)
    }
}
#[derive(Debug, thiserror::Error)]
pub enum PchanSteelErr {
    #[error(transparent)]
    SendError(#[from] SendError),
    #[error(transparent)]
    ReceiveError(#[from] ReceiveError),
    #[error("unknown gpr: {0}")]
    UnknownGpr(String),
    #[error(transparent)]
    UnhandledEmuIO(#[from] UnhandledIO),
    #[error("could not parse hex: {0}")]
    ParseHexError(#[from] ParseIntError),
    #[error("breakpoint not found: {}", hex(*.0))]
    BreakpointNotFound(u32),
}

impl From<PchanSteelErr> for SteelErr {
    fn from(value: PchanSteelErr) -> Self {
        match value {
            PchanSteelErr::SendError(_)
            | PchanSteelErr::ReceiveError(_)
            | PchanSteelErr::UnhandledEmuIO(_) => {
                SteelErr::new(SteelErrorKind::Io, format!("{value}"))
            }
            PchanSteelErr::UnknownGpr(_) | PchanSteelErr::ParseHexError(_) => {
                SteelErr::new(SteelErrorKind::Parse, format!("{value}"))
            }
            PchanSteelErr::BreakpointNotFound(_) => {
                SteelErr::new(SteelErrorKind::Generic, format!("{value}"))
            }
        }
    }
}

trait ErrorConvert {
    type Out;
    fn steel(self) -> Self::Out;
}

impl<T, E> ErrorConvert for Result<T, E>
where
    E: Into<PchanSteelErr>,
{
    type Out = Result<T, SteelErr>;

    fn steel(self) -> Self::Out {
        self.map_err(|err| err.into().into())
    }
}

type SteelResult<T> = Result<T, SteelErr>;

fn future<T, F: Send + 'static + Future<Output = Result<T, SteelErr>>>(
    conn: &ScriptConn,
    f: impl FnOnce(ScriptConn) -> F,
) -> SteelVal
where
    T: IntoSteelVal,
{
    let conn = conn.clone();
    SteelVal::FutureV(Gc::new(FutureResult::new(Box::pin({
        let f = f(conn.clone());
        async move { f.await?.into_steelval() }
    }))))
}

impl Default for SteelCtx {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SteelExecutor {
    wakers: Vec<(u64, Sender<()>)>,
    conn:   ScriptConn,
}

impl SteelExecutor {
    #[must_use]
    pub fn new(conn: ScriptConn) -> Self {
        SteelExecutor {
            wakers: vec![],
            conn,
        }
    }

    pub fn handle_call<EA: Allocator + Copy, RA: Allocator + Copy>(
        &mut self,
        emu: &mut Emu<EA>,
        runner: &mut Runner<RA>,
        msg: Call,
    ) {
        match msg {
            Call::Run => {
                runner.running = true;
            }
            Call::Pause => {
                runner.running = false;
            }
            Call::HardReset => {
                runner.frame_idx = 0;

                emu.gpu.wait_for_render_result();
                let bios_path = emu.boot.bios_path.clone();
                let alloc = emu.alloc;
                emu.gpu.reset_renderer();
                let mut new = Emu::new_in(alloc);
                new.set_bios_path(bios_path);
                new.load_bios(alloc).unwrap();
                new.cpu.jump_to_bios();
                new.tty.set_tracing();

                mem::swap(&mut new.gpu.conn, &mut emu.gpu.conn);
                new.spu.put_prod(emu.spu.take_prod());

                *emu = new;
            }
            Call::WaitFrames(n, waker) => {
                runner.running = true;
                if n == 0 {
                    waker.send(()).unwrap();
                }
                let wakeup = runner.frame_idx + u64::from(n);
                self.wakers.push((wakeup, waker));
            }
            Call::Gpr(tx) => {
                tx.send(emu.cpu.gpr.to_vec()).unwrap();
            }
            Call::GprSingle(gpr, tx) => tx.send(emu.cpu.gpr[gpr as usize]).unwrap(),
            Call::MemReadU32(MemReadCall(addr, tx)) => tx.send(emu.try_read(addr)).unwrap(),
            Call::Frame(tx) => tx.send(runner.frame_idx as isize).unwrap(),
            Call::AddBreakpoint(breakpoint) => {
                emu.dbg.breakpoints.insert(breakpoint.address, breakpoint);
            }
            Call::DelBreakpoint(address) => {
                emu.dbg.remove_breakpoint(address);
            }
            Call::SwitchBreakpoint(addr, val, tx) => match emu.dbg.breakpoints.get_mut(&addr) {
                Some(brk) => {
                    brk.enabled = val;
                    tx.send(Ok(())).unwrap();
                }
                None => tx
                    .send(Err(PchanSteelErr::BreakpointNotFound(addr)))
                    .unwrap(),
            },
        }
    }

    pub fn handle_step<EA: Allocator + Copy, RA: Allocator + Copy>(
        &mut self,
        emu: &mut Emu<EA>,
        runner: &mut Runner<RA>,
    ) {
        let idx = runner.frame_idx;
        self.wakers.retain(|(n, waker)| {
            if *n <= idx {
                runner.running = false;
                let _ = waker.send(());
                false
            } else {
                true
            }
        });
    }
}

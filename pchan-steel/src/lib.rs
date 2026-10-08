use core::alloc::Allocator;
use core::error::Error;
use core::mem;
use std::collections::HashMap;

use kanal::{ReceiveError, SendError, Sender};
use pchan_emu::Emu;
use pchan_emu::cpu::REG_STR;
use pchan_emu::run::Runner;
use pchan_utils::{AsyncChan, hex};
use steel::gc::Gc;
use steel::rerrs::ErrorKind as SteelErrorKind;
use steel::rvals::{FutureResult, IntoSteelVal};
use steel::steel_vm::builtin::BuiltInModule;
use steel::steel_vm::engine::Engine;
use steel::steel_vm::register_fn::RegisterFn;
use steel::{SteelErr, SteelVal};
use steel_derive::Steel;

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
    WaitFrames(u32, Sender<()>),
    Gpr(Sender<Vec<u32>>),
}

impl SteelCtx {
    #[must_use]
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let mut module = BuiltInModule::new("pchan/emu");
        let conn = ScriptConn {
            chan: kanal::bounded_async(16),
        };

        engine.register_type::<SteelU32>("u32");
        engine.register_type::<SteelGprMap>("pchan::SteelGprMap");

        module.register_fn("await", move |value: SteelVal| {
            if let SteelVal::FutureV(f) = value {
                let shared = f.unwrap().into_shared();
                smol::block_on(shared)
            } else {
                Ok(value)
            }
        });

        let c = conn.clone();
        module.register_fn("run", move || c.clone().send_sync(Call::Run).to_err());
        let c = conn.clone();
        module.register_fn("pause", move || c.send_sync(Call::Pause).to_err());
        let c = conn.clone();
        module.register_fn("hard-reset", move || c.send_sync(Call::HardReset).to_err());

        let c = conn.clone();
        module.register_fn("frames", move |n: u32| {
            future(&c, async move |c| {
                let (tx, rx) = kanal::bounded_async(0);
                c.send_async(Call::WaitFrames(n, tx.to_sync()))
                    .await
                    .to_err()?;
                rx.recv().await.to_err()?;
                Ok(())
            })
        });

        let c = conn.clone();
        module.register_fn("emu.cpu.gpr", move || {
            let (tx, rx) = kanal::bounded(0);
            c.send_sync(Call::Gpr(tx)).to_err()?;
            rx.recv().to_err().map(|gpr| {
                gpr.iter()
                    .copied()
                    .enumerate()
                    .map(|(gpr, value)| (REG_STR[gpr], SteelU32(value)))
                    .collect::<HashMap<_, _>>()
            })
        });

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

#[derive(Clone, derive_more::Debug)]
#[debug("{}", hex(self.0))]
struct SteelU32(u32);

impl steel::rvals::Custom for SteelU32 {
    fn fmt(&self) -> Option<core::result::Result<String, core::fmt::Error>> {
        Some(Ok(format!("{self:?}")))
    }

    fn into_serializable_steelval(&mut self) -> Option<steel::rvals::SerializableSteelVal> {
        Some(steel::rvals::SerializableSteelVal::IntV(self.0 as isize))
    }

    fn visit_equality(&self, _visitor: &mut steel::rvals::cycles::EqualityVisitor) {}

    fn equality_hint(&self, _other: &dyn steel::rvals::CustomType) -> bool {
        true
    }

    fn equality_hint_general(&self, other: &SteelVal) -> bool {
        matches!(other, SteelVal::IntV(num) if *num as u32 == self.0)
    }
}

#[derive(Clone, derive_more::Debug)]
struct SteelGprMap(HashMap<&'static str, SteelU32>);

impl steel::rvals::Custom for SteelGprMap {
    fn fmt(&self) -> Option<core::result::Result<String, core::fmt::Error>> {
        Some(Ok(format!("{:#?}", self.0)))
    }
}

#[derive(Debug, thiserror::Error)]
enum PchanSteelErr {
    #[error(transparent)]
    SendError(#[from] SendError),
    #[error(transparent)]
    ReceiveError(#[from] ReceiveError),
}

impl From<PchanSteelErr> for SteelErr {
    fn from(value: PchanSteelErr) -> Self {
        match value {
            PchanSteelErr::SendError(_) | PchanSteelErr::ReceiveError(_) => {
                SteelErr::new(SteelErrorKind::Io, format!("{value}"))
            }
        }
    }
}

trait SteelDiagnostic {
    type Out;
    fn io_err(self) -> Self::Out;
}

impl<T, E: Error> SteelDiagnostic for Result<T, E> {
    type Out = Result<T, SteelErr>;

    fn io_err(self) -> Self::Out {
        self.map_err(|err| SteelErr::new(SteelErrorKind::Io, format!("{err}")))
    }
}

trait ErrorConvert {
    type Out;
    fn to_err(self) -> Self::Out;
}

impl<T, E> ErrorConvert for Result<T, E>
where
    E: Into<PchanSteelErr>,
{
    type Out = Result<T, SteelErr>;

    fn to_err(self) -> Self::Out {
        self.map_err(|err| err.into().into())
    }
}

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
}

impl SteelExecutor {
    #[must_use]
    pub fn new() -> Self {
        SteelExecutor { wakers: vec![] }
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
        }
    }

    pub fn handle_step<EA: Allocator + Copy, RA: Allocator + Copy>(
        &mut self,
        emu: &mut Emu<EA>,
        runner: &mut Runner<RA>,
    ) {
        let event = self.wakers.iter().find(|(n, _)| *n == runner.frame_idx);
        if let Some((_, waker)) = event {
            let _ = waker.send(());
        }
    }
}

impl Default for SteelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

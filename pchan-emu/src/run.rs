use crate::Emu;
use crate::cpu::interp::{Interpreter, InterpreterResult};
use crate::dynarec_v2::{Dynarec, run_step};

#[derive(Debug)]
pub struct Runner {
    interpreter:     Interpreter,
    dynarec:         Option<Box<Dynarec>>,
    pub(crate) mode: RunnerMode,
    pub config:      RunnerConfig,
}

impl Default for Runner {
    fn default() -> Self {
        Self {
            interpreter: Default::default(),
            dynarec:     Some(Box::default()),
            mode:        Default::default(),
            config:      Default::default(),
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum RunnerMode {
    Dynarec,
    #[default]
    Interpreter,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct RunnerConfig {
    pub force_mode: Option<RunnerMode>,
}

impl Runner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: RunnerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn execute(&mut self, emu: &mut Emu) {
        loop {
            let mode = self.config.force_mode.unwrap_or(self.mode);
            match mode {
                RunnerMode::Dynarec => {
                    if let Some(dynarec) = self.dynarec.take() {
                        self.dynarec = Some(run_step(emu, dynarec));
                    }
                    return;
                }
                RunnerMode::Interpreter => match self.interpreter.run_instruction(emu) {
                    InterpreterResult::None => {}
                    _ => return,
                },
            }
        }
    }
}

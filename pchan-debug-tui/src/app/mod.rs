use std::sync::{Arc, RwLock};
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use flume::{Receiver, Sender};
use pchan_emu::Emu;
use pchan_emu::jit::JIT;
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType};
use ratatui::{DefaultTerminal, crossterm::event};

use crate::AppConfig;
use crate::app::first_time_setup::{FirstTimeSetup, FirstTimeSetupState};
use crate::app::modeline::{Command, Mode, Modeline, ModelineState};

pub mod first_time_setup;
pub mod modeline;

#[macro_export]
macro_rules! key {
    // Variant with no arguments, e.g. Enter, Esc
    ($code:ident, $kind:ident) => {
        ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code,
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            ..
        }
    };
    // Variant with arguments, e.g. Char('x'), Char(_)
    ($code:ident ( $($arg:tt)* ), $kind:ident) => {
        ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code($($arg)*),
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            ..
        }
    };
}

trait Component: StatefulWidget {
    type ComponentState;
    type ComponentSummary = ();
    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary>;
}

pub fn run(config: AppConfig, mut terminal: DefaultTerminal) -> Result<()> {
    let emu = Arc::new(RwLock::new(Emu::default()));
    let (config_tx, config_rx) = flume::unbounded();

    emu_thread(emu, config_rx)?;

    let mut app = App {};
    let mut app_state = AppState {
        app_config: config,
        exit: false,
        config_tx,
        error: None,

        focused: AppFocus::None,
        modeline_state: ModelineState { mode: Mode::Normal },
        first_time_setup_state: FirstTimeSetupState::default(),
    };

    if !app_state.app_config.initialized() {
        app_state.first_time_setup_state.bios_path_input_active = true;
        app_state.focused = AppFocus::FirstTimeSetupBiosInput;
    }

    loop {
        terminal.draw(|frame| frame.render_stateful_widget(app, frame.area(), &mut app_state))?;
        if let Ok(true) = event::poll(Duration::from_millis(0)) {
            let event = event::read()?;
            app.handle_input(event, &mut app_state)?;
        }
        if app_state.exit {
            return Ok(());
        }
    }
}

fn emu_thread(emu: Arc<RwLock<Emu>>, config_rx: Receiver<AppConfig>) -> Result<()> {
    std::thread::spawn(move || -> Result<()> {
        let mut jit = JIT::default();
        let Ok(mut config) = config_rx.recv() else {
            return Err(eyre!("emu thread received config with no bios path"));
        };
        {
            let mut emu = emu.write().unwrap();
            emu.boot.set_bios_path(config.bios_path.unwrap());
            emu.load_bios()?;
            emu.jump_to_bios();
        }
        loop {
            if let Ok(new_config) = config_rx.try_recv() {
                config = new_config;
            }
            emu.write().unwrap().step_jit(&mut jit)?;
        }
    });

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct App {}

pub struct AppState {
    app_config: AppConfig,
    config_tx: Sender<AppConfig>,
    exit: bool,
    error: Option<color_eyre::Report>,

    focused: AppFocus,
    modeline_state: ModelineState,
    first_time_setup_state: FirstTimeSetupState,
}

#[derive(Default, PartialEq, Eq)]
enum AppFocus {
    #[default]
    None,
    FirstTimeSetupBiosInput,
}

impl StatefulWidget for App {
    type State = AppState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let main = Block::bordered()
            .border_type(BorderType::Rounded)
            .title("🐗 Pーちゃん debugger");
        let main_area = main.inner(area);
        main.render(area, buf);
        let area = main_area;

        Modeline.render(area, buf, &mut state.modeline_state);

        match state.app_config.initialized() {
            false => {
                FirstTimeSetup.render(area, buf, &mut state.first_time_setup_state);
            }
            true => todo!(),
        }
    }
}

impl Component for App {
    type ComponentState = AppState;

    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        if state.focused == AppFocus::None {
            let cmd = match Modeline.handle_input(event.clone(), &mut state.modeline_state) {
                Ok(cmd) => cmd,
                Err(err) => {
                    state.error = Some(err);
                    return Ok(());
                }
            };
            match cmd {
                Command::None => {}
                Command::Quit => state.exit = true,
            };
        }
        match state.app_config.initialized() {
            // typing bios path and focused on the input
            false if state.focused == AppFocus::FirstTimeSetupBiosInput => {
                let action =
                    match FirstTimeSetup.handle_input(event, &mut state.first_time_setup_state) {
                        Ok(action) => action,
                        Err(err) => {
                            state.error = Some(err);
                            return Ok(());
                        }
                    };
                match action {
                    TypingAction::Escape => {
                        state.focused = AppFocus::None;
                    }
                    TypingAction::Pending => {}
                    TypingAction::Submit(path) => {
                        state.app_config.bios_path = Some(path);
                        state.config_tx.send(state.app_config.clone())?;
                    }
                }
            }

            // not focused
            false => {
                if let event::Event::Key(key!(Enter, Press)) = event {
                    state.focused = AppFocus::FirstTimeSetupBiosInput;
                    state.first_time_setup_state.bios_path_input_active = true;
                }
            }

            true => {}
            _ => {}
        }
        Ok(())
    }
}
pub enum TypingAction<T> {
    Escape,
    Pending,
    Submit(T),
}

pub struct MainPage;
pub struct MainPageState;

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use flume::{Receiver, Sender};
use pchan_emu::Emu;
use pchan_emu::jit::JIT;
use ratatui::crossterm::event::KeyEvent;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use ratatui::{DefaultTerminal, crossterm::event};
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use crate::AppConfig;

macro_rules! key {
    // Variant with no arguments, e.g. Enter, Esc
    ($code:ident, $kind:ident) => {
        KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code,
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            ..
        }
    };
    // Variant with arguments, e.g. Char('x'), Char(_)
    ($code:ident ( $($arg:tt)* ), $kind:ident) => {
        KeyEvent {
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

struct Modeline;
enum Mode {
    Normal,
    Command(ModeCommandState),
}
struct ModeCommandState {
    input: Input,
}
struct ModelineState {
    mode: Mode,
}
enum Command {
    None,
    Quit,
}

impl FromStr for Command {
    type Err = color_eyre::Report;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "q" | "quit" => Ok(Command::Quit),
            _ => Err(eyre!("invalid command")),
        }
    }
}

impl StatefulWidget for Modeline {
    type State = ModelineState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let [_, bottom] = Layout::vertical([Constraint::Min(1), Constraint::Max(1)]).areas(area);
        match &state.mode {
            Mode::Normal => {
                Paragraph::new("  NORM  ").render(bottom, buf);
            }
            Mode::Command(mode_command_state) => {
                let text = format!("  CMD: {}  ", mode_command_state.input.value());
                Paragraph::new(text).render(bottom, buf);
            }
        }
    }
}

impl Component for Modeline {
    type ComponentState = ModelineState;
    type ComponentSummary = Command;

    fn handle_input(&mut self, event: event::Event, state: &mut ModelineState) -> Result<Command> {
        if let Mode::Command(state) = &mut state.mode {
            state.input.handle_event(&event);
        }

        if let event::Event::Key(key_event) = event {
            match (&mut state.mode, key_event) {
                (Mode::Normal, key!(Char(':'), Press)) => {
                    state.mode = Mode::Command(ModeCommandState {
                        input: Input::default(),
                    });
                }
                (Mode::Command(cmd), key!(Enter, Press)) => {
                    let cmd = cmd.input.value();
                    let cmd = Command::from_str(cmd);
                    state.mode = Mode::Normal;
                    return cmd;
                }
                (Mode::Command(_), key!(Esc, Press)) => {
                    state.mode = Mode::Normal;
                }
                _ => {}
            }
        };

        Ok(Command::None)
    }
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

        if state.app_config.initialized() {
            state.config_tx.send(state.app_config.clone())?;
        }
        Ok(())
    }
}

struct FirstTimeSetup;
#[derive(Default)]
struct FirstTimeSetupState {
    bios_path_input: Input,
    bios_path_input_active: bool,
}

impl StatefulWidget for FirstTimeSetup {
    type State = FirstTimeSetupState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let [area] = Layout::horizontal([Constraint::Ratio(1, 3)])
            .flex(Flex::Center)
            .areas(area);
        let [header, bios_path_input] =
            Layout::vertical([Constraint::Max(4), Constraint::Length(3)])
                .flex(Flex::Center)
                .areas(area);
        Paragraph::new(
            "Welcome to the P-chan debugger!\n\nPlease provide the information required to run the debugger.",
        )
        .wrap(Wrap { trim: false })
        .centered()
        .render(header, buf);
        Paragraph::new(state.bios_path_input.value())
            .block(
                Block::bordered()
                    .title("path to bios file:")
                    .border_type(BorderType::Rounded)
                    .border_style(match state.bios_path_input_active {
                        true => Style::new().green(),
                        false => Style::new(),
                    }),
            )
            .render(bios_path_input, buf);
    }
}

enum TypingAction<T> {
    Escape,
    Pending,
    Submit(T),
}

impl Component for FirstTimeSetup {
    type ComponentState = FirstTimeSetupState;

    type ComponentSummary = TypingAction<PathBuf>;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        state.bios_path_input_active = true;
        state.bios_path_input.handle_event(&event);
        if let event::Event::Key(key) = event {
            match key {
                key!(Enter, Press) if !state.bios_path_input.value().is_empty() => {
                    state.bios_path_input_active = false;
                    let path = state.bios_path_input.value().to_string();
                    return Ok(TypingAction::Submit(path.into()));
                }
                _ => {}
            }
        }
        Ok(TypingAction::Pending)
    }
}

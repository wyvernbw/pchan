use std::borrow::Cow;
use std::hash::Hasher;
use std::hash::{DefaultHasher, Hash};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use flume::{Receiver, Sender};
use pchan_emu::Emu;
use pchan_emu::cpu::ops::Op;
use pchan_emu::dynarec::{FetchParams, FetchSummary};
use pchan_emu::jit::JIT;
use pchan_utils::hex;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::style::Stylize;
use ratatui::widgets::{Block, BorderType, List, ListState, Padding, Paragraph, Wrap};
use ratatui::{DefaultTerminal, crossterm::event};

use crate::AppConfig;
use crate::app::component::{Component, ComponentWidget};
use crate::app::first_time_setup::{FirstTimeSetup, FirstTimeSetupState};
use crate::app::focus::{FocusProp, FocusProvider};
use crate::app::modeline::{Command, Mode, Modeline, ModelineState};

pub mod component;
pub mod first_time_setup;
pub mod focus;
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

pub fn run(config: AppConfig, mut terminal: DefaultTerminal) -> Result<()> {
    let emu = Arc::new(RwLock::new(Emu::default()));
    let (config_tx, config_rx) = flume::unbounded();
    let (emu_cmd_tx, emu_cmd_rx) = flume::unbounded();
    let (emu_info_tx, emu_info_rx) = flume::unbounded();
    emu_cmd_tx.send(EmuCmd::RunFetch)?;

    emu_thread(emu, config_rx, emu_cmd_rx, emu_info_tx)?;

    let mut app_state = AppState {
        screen: if config.initialized() {
            Screen::Main
        } else {
            Screen::FirstTimeSetup
        },
        app_config: config,
        exit: false,
        config_tx,
        error: None,
        focus: FocusProvider::default(),

        modeline_state: ModelineState { mode: Mode::Normal },
        first_time_setup_state: FirstTimeSetupState::default(),
        main_page_state: MainPageState {
            fetch_summary: None,
            emu_state: EmuState::Uninitialized,
            ops_list_state: OpsListState::default(),
            focus: MainPageFocus::default(),
        },
    };

    loop {
        let mut app = App(app_state.focus.props());
        terminal.draw(|frame| {
            let result = app
                .clone()
                .render(frame.area(), frame.buffer_mut(), &mut app_state);
            if let Err(err) = result {
                app_state.error = Some(err);
            }
        })?;
        if let Ok(true) = event::poll(Duration::from_millis(0)) {
            let event = event::read()?;
            let result = app.handle_input(event, &mut app_state);
            if let Err(err) = result {
                app_state.error = Some(err);
            }
        }
        if let Ok(info) = emu_info_rx.try_recv() {
            match info {
                EmuInfo::Fetch(fetch_summary) => {
                    app_state.main_page_state.fetch_summary = Some(fetch_summary)
                }
                EmuInfo::StateUpdate(emu_state) => app_state.main_page_state.emu_state = emu_state,
            }
        }
        if app_state.exit {
            return Ok(());
        }
        app_state.focus.process();
    }
}

pub enum EmuCmd {
    RunFetch,
}

pub enum EmuState {
    Uninitialized,
    Error(color_eyre::Report),
    WaitingForConfig,
    SettingUp,
    Running,
}

pub enum EmuInfo {
    StateUpdate(EmuState),
    Fetch(FetchSummary),
}

fn emu_thread(
    emu: Arc<RwLock<Emu>>,
    config_rx: Receiver<AppConfig>,
    emu_cmd_rx: Receiver<EmuCmd>,
    emu_info_tx: Sender<EmuInfo>,
) -> Result<()> {
    emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Uninitialized))?;
    std::thread::spawn(move || -> Result<()> {
        let mut jit = JIT::default();
        emu_info_tx.send(EmuInfo::StateUpdate(EmuState::WaitingForConfig))?;
        let Ok(mut config) = config_rx.recv() else {
            emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Error(eyre!(
                "emu thread received config with no bios path"
            ))))?;
            return Err(eyre!("emu thread received config with no bios path"));
        };
        emu_info_tx.send(EmuInfo::StateUpdate(EmuState::SettingUp))?;
        {
            let mut emu = emu.write().unwrap();
            emu.boot.set_bios_path(config.bios_path.unwrap());
            match emu.load_bios() {
                Ok(_) => {}
                Err(err) => {
                    emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Error(err)))?;
                    return Ok(());
                }
            }
            emu.jump_to_bios();
        }
        emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Running))?;
        loop {
            if let Ok(new_config) = config_rx.try_recv() {
                config = new_config;
            }
            if let Ok(emu_cmd) = emu_cmd_rx.try_recv() {
                let emu = emu.write().unwrap();
                match emu_cmd {
                    EmuCmd::RunFetch => {
                        let initial_address = emu.cpu.pc;
                        let ptr_type = jit.pointer_type();
                        let mut fetch_result =
                            emu.fetch(FetchParams::builder().pc(initial_address).build())?;
                        emu_info_tx.send(EmuInfo::Fetch(fetch_result))?;
                    }
                }
            }
        }
    });

    Ok(())
}

pub enum Screen {
    Main,
    FirstTimeSetup,
}

#[derive(Debug, Clone)]
pub struct App(FocusProp<App>);

pub struct AppState {
    app_config: AppConfig,
    screen: Screen,
    config_tx: Sender<AppConfig>,
    exit: bool,
    error: Option<color_eyre::Report>,

    modeline_state: ModelineState,
    first_time_setup_state: FirstTimeSetupState,
    main_page_state: MainPageState,
    focus: FocusProvider,
}

impl Component for App {
    type ComponentState = AppState;

    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        let cmd = Modeline(self.0.clone().prop())
            .handle_input(event.clone(), &mut state.modeline_state)?;
        match cmd {
            Command::Escape => {}
            Command::None => {}
            Command::Quit => state.exit = true,
        };

        match state.screen {
            Screen::FirstTimeSetup => {
                let action = FirstTimeSetup(self.0.prop())
                    .handle_input(event, &mut state.first_time_setup_state)?;
                match action {
                    TypingAction::Escape => {}
                    TypingAction::Pending => {}
                    TypingAction::Submit(path) => {
                        state.app_config.bios_path = Some(path);
                        state.config_tx.send(state.app_config.clone())?;
                    }
                    TypingAction::Enter => {}
                }
            }
            Screen::Main => {
                MainPage(self.0.prop()).handle_input(event, &mut state.main_page_state)?;
            }
        }

        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let main = Block::bordered()
            .border_type(BorderType::Rounded)
            .title("🐗 Pーちゃん debugger");
        let main_area = main.inner(area);
        main.render(area, buf);
        let area = main_area;

        match state.app_config.initialized() {
            false => {
                FirstTimeSetup(self.0.prop()).render(
                    area,
                    buf,
                    &mut state.first_time_setup_state,
                )?;
            }
            true => MainPage(self.0.prop()).render(area, buf, &mut state.main_page_state)?,
        }

        Modeline(self.0.prop()).render(area, buf, &mut state.modeline_state)?;

        Ok(())
    }
}
pub enum TypingAction<T> {
    Escape,
    Enter,
    Pending,
    Submit(T),
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MainPageFocus {
    None,
    #[default]
    OpsList,
}

pub struct MainPage(FocusProp<MainPage>);
pub struct MainPageState {
    emu_state: EmuState,
    fetch_summary: Option<FetchSummary>,
    ops_list_state: OpsListState,
    focus: MainPageFocus,
}

pub struct OpsList<'a>(&'a EmuState);
#[derive(Debug, Clone)]
pub struct OpsListData {
    ops: Vec<Line<'static>>,
    hash: u64,
}
#[derive(Default)]
pub struct OpsListState {
    list_state: ListState,
    data: Option<OpsListData>,
}

impl OpsListData {
    pub fn from_fetch(summary: &FetchSummary) -> Self {
        let decoded_ops = summary
            .cfg
            .node_weights()
            .map(|node| {
                summary
                    .ops_for(node)
                    .iter()
                    .enumerate()
                    .map(|(offset, op)| (node.ops.0.start + offset as u32 * 4, op))
                    .map(|(address, op)| {
                        Line::from(vec![
                            format!("{}   ", hex(address)).dim(),
                            if op.is_block_boundary().is_some() {
                                format!("{}", op).red()
                            } else {
                                format!("{}", op).into()
                            },
                        ])
                    })
                    .collect::<Vec<Line>>()
            })
            .intersperse(vec![Line::from("...")])
            .flatten()
            .collect::<Vec<_>>();
        let mut hasher = DefaultHasher::new();
        decoded_ops.hash(&mut hasher);
        Self {
            ops: decoded_ops,
            hash: hasher.finish(),
        }
    }
}

impl<'a> Component for OpsList<'a> {
    type ComponentState = OpsListState;

    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let OpsList(emu_state) = self;
        match &state.data {
            Some(data) => {
                StatefulWidget::render(
                    List::new(data.ops.clone()),
                    area,
                    buf,
                    &mut state.list_state,
                );
                Ok(())
            }
            None => {
                let [centered] = Layout::vertical([Constraint::Max(4)])
                    .flex(Flex::Center)
                    .areas(area);
                Paragraph::new(match emu_state {
                    EmuState::Error(err) => Cow::Owned(format!("何もない\nError: {}", err)),
                    EmuState::Running => Cow::Borrowed("何もない\nThe emulator is running..."),
                    _ => Cow::Borrowed("何もない\nThe emulator is not running"),
                })
                .wrap(Wrap { trim: false })
                .centered()
                .render(centered, buf);
                Ok(())
            }
        }
    }
}

impl Component for MainPage {
    type ComponentState = MainPageState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let [ops] = Layout::horizontal([Constraint::Ratio(1, 3)]).areas(area);
        let ops_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1))
            .title("OPコード");
        let ops_inner = ops_block.inner(ops);
        ops_block.render(ops, buf);
        let ops = ops_inner;

        match (&mut state.ops_list_state.data, &mut state.fetch_summary) {
            (None, None) => {}
            (None, Some(fetch_summary)) => {
                state.ops_list_state.data = Some(OpsListData::from_fetch(fetch_summary));
            }
            (Some(_), None) => {} // unreachable in practice (probably)
            (Some(old), Some(new)) => {
                let mut hasher = DefaultHasher::new();
                new.decoded_ops.hash(&mut hasher);
                let new_hash = hasher.finish();
                if old.hash != new_hash {
                    state.ops_list_state.data = Some(OpsListData::from_fetch(new));
                }
            }
        };
        OpsList(&state.emu_state).render(ops, buf, &mut state.ops_list_state)?;
        Ok(())
    }

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        Ok(())
    }
}

use std::borrow::Cow;
use std::hash::Hasher;
use std::hash::{DefaultHasher, Hash};
use std::ops::Shr;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use flume::{Receiver, Sender};
use pchan_emu::Emu;
use pchan_emu::cpu::ops::Op;
use pchan_emu::cpu::reg_str;
use pchan_emu::dynarec::pipeline::{EmuDynarecPipeline, EmuDynarecPipelineReport};
use pchan_emu::dynarec::{FetchParams, FetchSummary};
use pchan_emu::jit::JIT;
use pchan_utils::hex;
use ratatui::crossterm::event::KeyModifiers;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::style::Stylize;
use ratatui::widgets::{
    Block, BorderType, Gauge, LineGauge, List, ListState, Padding, Paragraph, Row, Table,
    TableState, Tabs, Wrap,
};
use ratatui::{DefaultTerminal, crossterm::event};
use strum::IntoEnumIterator;

use crate::AppConfig;
use crate::app::component::Component;
use crate::app::first_time_setup::{FirstTimeSetup, FirstTimeSetupState};
use crate::app::focus::{Focus, FocusProp, FocusProvider};
use crate::app::modeline::{Command, Mode, Modeline, ModelineState};
use crate::utils::InsertBetweenExt;

pub mod component;
pub mod first_time_setup;
pub mod focus;
pub mod modeline;

pub fn centered(area: Rect, vertical: Constraint) -> Rect {
    let [centered] = Layout::vertical([vertical]).flex(Flex::Center).areas(area);
    centered
}

#[macro_export]
macro_rules! key {
    // Variant with no arguments, e.g. Enter, Esc
    ($code:ident, $kind:ident) => {
        ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code,
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            modifiers: ratatui::crossterm::event::KeyModifiers::NONE,
            ..
        }
    };
    // Variant with arguments, e.g. Char('x'), Char(_)
    ($code:ident ( $($arg:tt)* ), $kind:ident) => {
        ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code($($arg)*),
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            modifiers: ratatui::crossterm::event::KeyModifiers::NONE,
            ..
        }
    };
    // Variant with no arguments and optional modifiers
    ($code:ident, $kind:ident, $mods:expr ) => {
        ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code,
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            modifiers: $mods,
            ..
        }
    };
    ($code:ident ( $($arg:tt)* ), $kind:ident, $mods:pat) => {
        ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code($($arg)*),
            kind: ratatui::crossterm::event::KeyEventKind::$kind,
            modifiers: $mods,
            ..
        }
    };
}

pub fn run(config: AppConfig, mut terminal: DefaultTerminal) -> Result<()> {
    let emu = Arc::new(RwLock::new(Emu::default()));
    let (config_tx, config_rx) = flume::unbounded();
    let (emu_cmd_tx, emu_cmd_rx) = flume::unbounded();
    let (emu_info_tx, emu_info_rx) = flume::unbounded();

    emu_thread(emu.clone(), config_rx, emu_cmd_rx, emu_info_tx)?;

    let mut app_state = AppState {
        focus: if config.initialized() {
            FocusProvider::new(Some(OpsList::as_focus()))
        } else {
            FocusProvider::new(Some(FirstTimeSetup::as_focus()))
        },
        screen: if config.initialized() {
            config_tx.send(config.clone())?;
            Screen::Main
        } else {
            Screen::FirstTimeSetup
        },
        app_config: config,
        exit: false,
        config_tx,
        error: None,
        emu: None,
        pipeline_report: EmuDynarecPipeline::new().report(),

        modeline_state: ModelineState { mode: Mode::Normal },
        first_time_setup_state: FirstTimeSetupState::default(),
        main_page_state: MainPageState {
            fetch_summary: None,
            emu_state: EmuState::Uninitialized.into(),
            cpu_inspector_state: CpuInspectorState {
                emu: None,
                current_tab: CpuInspectorTab::Cpu,
                cpu_tab_state: CpuTabState {
                    table_state: TableState::default(),
                    show_zero: true,
                },
            },
            ops_list_state: OpsListState::default(),
            focus: MainPageFocus::default(),
            emu: None,
            emu_cmd_tx,
            action_state: ActionState {
                chord: ActionChord::None,
            },
            memory_inspector_state: MemoryInspectorState { emu: emu.clone() },
        },
    };

    loop {
        let mut app = App(app_state.focus.props());
        if let Ok(true) = event::poll(Duration::from_millis(0)) {
            let event = event::read()?;
            let result = app.handle_input(event, &mut app_state);
            if let Err(err) = result {
                app_state.error = Some(err);
            }
        }
        if let Ok(info) = emu_info_rx.try_recv() {
            match info {
                EmuInfo::PipelineUpdate(report) => {
                    app_state.pipeline_report = report;
                }
                EmuInfo::Ref(emu) => {
                    app_state.emu = Some(emu.clone());
                    app_state.main_page_state.emu = Some(emu.clone());
                    app_state.main_page_state.cpu_inspector_state.emu = Some(emu.clone());
                }
                EmuInfo::Fetch(fetch_summary) => {
                    app_state.main_page_state.fetch_summary = Some(fetch_summary)
                }
                EmuInfo::StateUpdate(emu_state) => {
                    app_state.main_page_state.emu_state = emu_state.into()
                }
            }
        }
        if app_state.exit {
            return Ok(());
        }
        app_state.focus.process();
        terminal.draw(|frame| {
            let result = app
                .clone()
                .render(frame.area(), frame.buffer_mut(), &mut app_state);
            if let Err(err) = result {
                app_state.error = Some(err);
            }
        })?;
        std::thread::sleep(Duration::from_secs_f32(1.0 / 120.0));
    }
}

pub enum EmuCmd {
    StepJit,
    Run,
}

pub enum EmuState {
    Uninitialized,
    Error(color_eyre::Report),
    WaitingForConfig,
    SettingUp,
    Running,
}

pub enum EmuInfo {
    PipelineUpdate(EmuDynarecPipelineReport),
    StateUpdate(EmuState),
    Fetch(FetchSummary),
    Ref(Arc<RwLock<Emu>>),
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
        emu_info_tx.send(EmuInfo::Ref(emu.clone()))?;

        let mut pipeline = EmuDynarecPipeline::new();

        loop {
            if let Ok(new_config) = config_rx.try_recv() {
                config = new_config;
            }
            match emu_cmd_rx.try_recv() {
                Ok(EmuCmd::StepJit) => {
                    let mut emu = emu.write().unwrap();
                    let pc = emu.cpu.pc;
                    pipeline = pipeline.step(&mut emu, &mut jit, pc)?;
                    if let Some(fetch) = pipeline.as_fetch() {
                        emu_info_tx.send(EmuInfo::Fetch(fetch.clone()))?;
                    }
                    emu_info_tx.send(EmuInfo::PipelineUpdate(pipeline.report()))?;
                }
                Ok(EmuCmd::Run) => {
                    let mut emu = emu.write().unwrap();
                    let pc = emu.cpu.pc;
                    pipeline = pipeline.run(&mut emu, &mut jit, pc)?;
                    emu_info_tx.send(EmuInfo::PipelineUpdate(pipeline.report()))?;
                }
                Err(_) => {}
            }
        }
    });

    Ok(())
}

#[derive(Debug, Clone, Copy)]
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
    emu: Option<Arc<RwLock<Emu>>>,
    pipeline_report: EmuDynarecPipelineReport,

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
                        confy::store("pchan-debugger", "config", state.app_config.clone())?;
                        state.screen = Screen::Main;
                        self.0.prop::<OpsList>().push_focus();
                        state.config_tx.send(state.app_config.clone())?;
                    }
                    TypingAction::Enter => {}
                }
            }
            Screen::Main => {
                MainPage(self.0.prop(), &state.pipeline_report)
                    .handle_input(event, &mut state.main_page_state)?;
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

        match state.screen {
            Screen::FirstTimeSetup => {
                FirstTimeSetup(self.0.prop()).render(
                    area,
                    buf,
                    &mut state.first_time_setup_state,
                )?;
            }
            Screen::Main => MainPage(self.0.prop(), &state.pipeline_report).render(
                area,
                buf,
                &mut state.main_page_state,
            )?,
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

pub struct MainPage<'a>(FocusProp<MainPage<'a>>, &'a EmuDynarecPipelineReport);
pub struct MainPageState {
    emu_state: Arc<EmuState>,
    emu: Option<Arc<RwLock<Emu>>>,
    emu_cmd_tx: Sender<EmuCmd>,
    fetch_summary: Option<FetchSummary>,
    ops_list_state: OpsListState,
    cpu_inspector_state: CpuInspectorState,
    focus: MainPageFocus,
    action_state: ActionState,
    memory_inspector_state: MemoryInspectorState,
}

pub struct OpsList(Arc<EmuState>, FocusProp<OpsList>);
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
        let mut decoded_ops = summary
            .cfg
            .node_weights()
            .flat_map(|node| {
                summary
                    .ops_for(node)
                    .iter()
                    .enumerate()
                    .map(|(offset, op)| (node.address + offset as u32 * 4, op))
            })
            .collect::<Vec<_>>();
        decoded_ops.sort_by(|a, b| a.0.cmp(&b.0));
        decoded_ops.dedup_by(|a, b| a.0 == b.0);
        let decoded_ops = decoded_ops
            .iter()
            .map(|(address, op)| {
                (
                    address,
                    Line::from(vec![
                        format!("{}   ", hex(*address)).dim(),
                        if op.is_block_boundary().is_some() {
                            format!("{}", op).red()
                        } else {
                            format!("{}", op).into()
                        },
                    ])
                    .style(if address.shr(2) % 2u32 == 0u32 {
                        Style::new().fg(Color::Rgb(250 - 15, 250 - 15, 250 - 15))
                    } else {
                        Style::new()
                    }),
                )
            })
            .insert_between(
                |a, b| a.0.abs_diff(*b.0).ge(&8),
                || (&0u32, Line::from("...").style(Style::new().dim())),
            )
            .map(|(_, line)| line)
            .collect::<Vec<Line>>();
        let mut hasher = DefaultHasher::new();
        decoded_ops.hash(&mut hasher);
        Self {
            ops: decoded_ops,
            hash: hasher.finish(),
        }
    }
}

impl Focus for OpsList {
    fn focus_next() -> Option<focus::FocusId> {
        Some(CpuInspector::as_focus())
    }
}

impl Component for OpsList {
    type ComponentState = OpsListState;

    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        if !self.1.is_focused() {
            return Ok(());
        }
        let screen_height = ratatui::crossterm::terminal::size()?.1;
        match event {
            event::Event::Key(key!(Char('g'), Press)) => {
                state.list_state.select_first();
            }
            event::Event::Key(key!(Char('G'), Press, KeyModifiers::SHIFT)) => {
                state.list_state.select_last();
            }
            event::Event::Key(key!(Char('j'), Press)) => {
                state.list_state.select_next();
            }
            event::Event::Key(key!(Char('k'), Press)) => {
                state.list_state.select_previous();
            }
            event::Event::Key(key!(Char('J'), Press, KeyModifiers::SHIFT)) => {
                let selected =
                    state.list_state.selected().unwrap_or(0) + screen_height as usize - 4;
                state.list_state.select(Some(selected));
            }
            event::Event::Key(key!(Char('K'), Press, KeyModifiers::SHIFT)) => {
                let selected = state
                    .list_state
                    .selected()
                    .unwrap_or(0)
                    .saturating_sub(screen_height as usize - 4);
                state.list_state.select(Some(selected));
            }
            event::Event::Key(key!(Char('h'), Press)) => {
                self.1.focus_prev();
            }
            event::Event::Key(key!(Char('l'), Press)) => {
                self.1.focus_next();
            }
            _ => {}
        };
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let OpsList(emu_state, focus) = self;
        match &state.data {
            Some(data) => {
                StatefulWidget::render(
                    List::new(data.ops.clone()).highlight_style(Style::new().black().on_yellow()),
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
                Paragraph::new(match emu_state.as_ref() {
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

impl<'a> Component for MainPage<'a> {
    type ComponentState = MainPageState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let [ops, mid, memory] =
            Layout::horizontal([Constraint::Ratio(1, 3)].repeat(3)).areas(area);
        let ops_focus = self.0.prop::<OpsList>();
        let ops_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1))
            .border_style(ops_focus.style())
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
        OpsList(state.emu_state.clone(), ops_focus).render(ops, buf, &mut state.ops_list_state)?;

        let [cpu_inspector, actions] =
            Layout::vertical([Constraint::Ratio(2, 3), Constraint::Ratio(1, 3)])
                .spacing(1)
                .areas(mid);
        let cpu_inspector_focus = self.0.prop::<CpuInspector>();
        let cpu_inspector_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(cpu_inspector_focus.style())
            .title_bottom(" <Tab> Cycle ")
            .title("エミュ CPU");
        let cpu_inspector_inner = cpu_inspector_block.inner(cpu_inspector);
        cpu_inspector_block.render(cpu_inspector, buf);
        let cpu_inspector = cpu_inspector_inner;
        CpuInspector(cpu_inspector_focus).render(
            cpu_inspector,
            buf,
            &mut state.cpu_inspector_state,
        )?;

        let actions_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::uniform(1))
            .title("Actions");
        let actions_block_inner = actions_block.inner(actions);
        actions_block.render(actions, buf);
        Actions(&state.emu_cmd_tx, self.1).render(
            actions_block_inner,
            buf,
            &mut state.action_state,
        )?;

        let memory_inspector_focus = self.0.prop();
        let memory_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .style(memory_inspector_focus.style())
            .title("Memory");
        let memory_block_inner = memory_block.inner(memory);
        memory_block.render(memory, buf);
        MemoryInspector(memory_inspector_focus).render(
            memory_block_inner,
            buf,
            &mut state.memory_inspector_state,
        )?;

        Ok(())
    }

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        let ops_focus = self.0.prop::<OpsList>();
        OpsList(state.emu_state.clone(), ops_focus)
            .handle_input(event.clone(), &mut state.ops_list_state)?;
        CpuInspector(self.0.prop()).handle_input(event.clone(), &mut state.cpu_inspector_state)?;
        Actions(&state.emu_cmd_tx, self.1).handle_input(event.clone(), &mut state.action_state)?;
        MemoryInspector(self.0.prop()).handle_input(event, &mut state.memory_inspector_state)?;
        Ok(())
    }
}

pub struct CpuInspector(FocusProp<Self>);
pub struct CpuInspectorState {
    emu: Option<Arc<RwLock<Emu>>>,
    current_tab: CpuInspectorTab,
    cpu_tab_state: CpuTabState,
}

impl Focus for CpuInspector {
    fn focus_down() -> Option<focus::FocusId> {
        None
    }
    fn focus_up() -> Option<focus::FocusId> {
        None
    }
    fn focus_left() -> Option<focus::FocusId> {
        Some(OpsList::as_focus())
    }
    fn focus_right() -> Option<focus::FocusId> {
        Some(MemoryInspector::as_focus())
    }
}

impl Component for CpuInspector {
    type ComponentState = CpuInspectorState;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        if !self.0.is_focused() {
            return Ok(());
        }
        match event {
            event::Event::Key(key!(Tab, Press)) => {
                state.current_tab = state.current_tab.next();
            }
            event::Event::Key(key!(Char('h'), Press)) => {
                self.0.focus_left();
            }
            event::Event::Key(key!(Char('j'), Press)) => {
                self.0.focus_down();
            }
            event::Event::Key(key!(Char('k'), Press)) => {
                self.0.focus_up();
            }
            event::Event::Key(key!(Char('l'), Press)) => {
                self.0.focus_right();
            }
            _ => {}
        };
        let Some(emu) = state.emu.as_deref() else {
            return Ok(());
        };
        match state.current_tab {
            CpuInspectorTab::Cpu => {
                CpuTab(emu).handle_input(event, &mut state.cpu_tab_state)?;
            }
            CpuInspectorTab::Cop0 => todo!(),
            CpuInspectorTab::Gte => todo!(),
        }
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let emu = match state.emu.as_deref() {
            Some(emu) => emu,
            None => {
                let [centered] = Layout::vertical([Constraint::Max(4)])
                    .flex(Flex::Center)
                    .areas(area);
                Paragraph::new("何もない\nEmulator is not running")
                    .wrap(Wrap { trim: false })
                    .centered()
                    .render(centered, buf);
                return Ok(());
            }
        };

        let [tabs_area, tabs_inner] =
            Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(area);
        Tabs::new(CpuInspectorTab::iter().map(|el| el.to_string()))
            .style(Style::default().white())
            .highlight_style(Style::default().yellow())
            .select(state.current_tab as usize)
            .divider(symbols::border::PLAIN.vertical_left)
            .render(tabs_area, buf);
        match state.current_tab {
            CpuInspectorTab::Cpu => {
                CpuTab(emu).render(tabs_inner, buf, &mut state.cpu_tab_state)?
            }
            CpuInspectorTab::Cop0 => todo!(),
            CpuInspectorTab::Gte => todo!(),
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, derive_more::Display, strum::EnumIter)]
enum CpuInspectorTab {
    #[display("CPU")]
    Cpu,
    #[display("COP0")]
    Cop0,
    #[display("GTE")]
    Gte,
}

impl CpuInspectorTab {
    pub fn next(&self) -> Self {
        match self {
            CpuInspectorTab::Cpu => CpuInspectorTab::Cop0,
            CpuInspectorTab::Cop0 => CpuInspectorTab::Gte,
            CpuInspectorTab::Gte => CpuInspectorTab::Cpu,
        }
    }
}

pub struct CpuTab<'a>(&'a RwLock<Emu>);
pub struct CpuTabState {
    table_state: TableState,
    show_zero: bool,
}

impl<'a> Component for CpuTab<'a> {
    type ComponentState = CpuTabState;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        match event {
            event::Event::Key(key!(Char('j'), Press)) => {
                state.table_state.select_next();
            }
            event::Event::Key(key!(Char('k'), Press)) => {
                state.table_state.select_previous();
            }
            event::Event::Key(key!(Char('s'), Press)) => {
                state.show_zero = !state.show_zero;
            }
            _ => {}
        }
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let emu = self.0.read().unwrap();
        let rows = emu
            .cpu
            .gpr
            .iter()
            .enumerate()
            .filter(|(_, value)| if !state.show_zero { **value != 0 } else { true })
            .map(|(reg, value)| {
                Row::new([
                    Cow::Owned(format!("${}", reg_str(reg as u8))),
                    Cow::Borrowed(hex(*value)),
                ])
                .style(if *value == 0 {
                    Style::new().dim()
                } else {
                    Style::new()
                })
            })
            .collect::<Vec<_>>();

        if rows.is_empty() {
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .title_bottom("<s> Show zero registers");
            let block_inner = block.inner(area);
            block.render(area, buf);
            Paragraph::new("ゼロ ー All registers are zero.")
                .wrap(Wrap { trim: false })
                .centered()
                .render(centered(block_inner, Constraint::Max(2)), buf);
            return Ok(());
        }

        StatefulWidget::render(
            Table::new(rows, [Constraint::Max(8), Constraint::Fill(1)])
                .row_highlight_style(Style::new().black().on_yellow())
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .title_bottom("<s> Show zero registers"),
                )
                .header(Row::new(["レジスタ", "Value"])),
            area,
            buf,
            &mut state.table_state,
        );

        Ok(())
    }
}

pub struct Actions<'a>(&'a Sender<EmuCmd>, &'a EmuDynarecPipelineReport);
pub struct ActionState {
    chord: ActionChord,
}

pub enum ActionChord {
    None,
    Breakpoint,
}

impl<'a> Component for Actions<'a> {
    type ComponentState = ActionState;
    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        match (&state.chord, event) {
            (ActionChord::None, event::Event::Key(key!(Char('n'), Press))) => {
                self.0.send(EmuCmd::StepJit)?;
            }
            (ActionChord::None, event::Event::Key(key!(Char('r'), Press))) => {
                self.0.send(EmuCmd::Run)?;
            }
            (ActionChord::None, event::Event::Key(key!(Char('b'), Press))) => {
                state.chord = ActionChord::Breakpoint;
            }
            (ActionChord::Breakpoint, event::Event::Key(key!(Char('a'), Press))) => {
                todo!("add breakpoint")
            }
            (ActionChord::Breakpoint, event::Event::Key(key!(Char('d'), Press))) => {
                todo!("delete breakpoint")
            }
            (ActionChord::Breakpoint, event::Event::Key(_)) => {
                state.chord = ActionChord::None;
            }
            _ => {}
        }
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let [actions, report, stage_list] =
            Layout::vertical([Constraint::Max(6), Constraint::Max(2), Constraint::Max(5)])
                .areas(area);
        Paragraph::new(format!(
            "<n>  next: {} 
            <r>  run 
            <ba> add breakpoint
            <bd> delete breakpoint
            ",
            self.1.next
        ))
        .wrap(Wrap { trim: true })
        .render(actions, buf);

        LineGauge::default()
            .block(
                Block::new()
                    .title(format!("Emu Stage -> {}", self.1.current))
                    .title_style(Style::new().bold()),
            )
            .unfilled_style(Style::new().dark_gray())
            .filled_style(Style::new().green().on_black())
            .line_set(symbols::line::THICK)
            .ratio(self.1.progress as f64 / { EmuDynarecPipeline::max_progress() as f64 })
            .render(report, buf);

        let progress = ["ready", "fetched", "emitted", "called", "cached"]
            .into_iter()
            .enumerate()
            .map(|(i, state)| (i, format!(" - {}", state)))
            .map(|(i, state)| {
                Line::from(state).style(if i <= self.1.progress {
                    Style::new().white()
                } else {
                    Style::new().dark_gray()
                })
            });
        Text::from_iter(progress).render(stage_list, buf);
        Ok(())
    }
}

pub struct MemoryInspector(FocusProp<Self>);
pub struct MemoryInspectorState {
    emu: Arc<RwLock<Emu>>,
}

impl Focus for MemoryInspector {
    fn focus_left() -> Option<focus::FocusId> {
        Self::focus_prev()
    }

    fn focus_right() -> Option<focus::FocusId> {
        Self::focus_next()
    }

    fn focus_next() -> Option<focus::FocusId> {
        Some(OpsList::as_focus())
    }

    fn focus_prev() -> Option<focus::FocusId> {
        Some(CpuInspector::as_focus())
    }
}

impl Component for MemoryInspector {
    type ComponentState = MemoryInspectorState;

    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        if !self.0.is_focused() {
            return Ok(());
        }
        match event {
            event::Event::Key(key!(Char('h'), Press)) => {
                self.0.focus_left();
            }
            event::Event::Key(key!(Char('j'), Press)) => {
                self.0.focus_down();
            }
            event::Event::Key(key!(Char('k'), Press)) => {
                self.0.focus_up();
            }
            event::Event::Key(key!(Char('l'), Press)) => {
                self.0.focus_right();
            }
            _ => {}
        };
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        Ok(())
    }
}

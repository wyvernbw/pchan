use std::borrow::Cow;
use std::collections::HashSet;
use std::hash::Hasher;
use std::hash::{DefaultHasher, Hash};
use std::ops::Shr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use circular_buffer::CircularBuffer;
use color_eyre::eyre::{Result, eyre};
use flume::{Receiver, Sender};
use pchan_emu::bootloader::Bootloader;
use pchan_emu::cpu::ops::{BoundaryType, Op};
use pchan_emu::cpu::{Cpu, reg_str};
use pchan_emu::dynarec::FetchSummary;
use pchan_emu::dynarec::pipeline::EmuDynarecPipelineReport;
use pchan_emu::dynarec_v2::emitters::DynarecOp;
use pchan_emu::dynarec_v2::{FetchedOp, PipelineV2, PipelineV2Stage};
use pchan_emu::io::IO;
use pchan_emu::{Emu, dynarec_v2};
use pchan_utils::{IgnorePoison, hex};
use ratatui::crossterm::event::KeyModifiers;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::style::Stylize;
use ratatui::widgets::{
    Block, BorderType, Clear, LineGauge, List, ListState, Padding, Paragraph, Row, Table,
    TableState, Tabs, Wrap,
};
use ratatui::{DefaultTerminal, crossterm::event};
use smol::Executor;
use strum::{EnumCount, IntoEnumIterator};
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use crate::AppConfig;
use crate::app::component::Component;
use crate::app::first_time_setup::{FirstTimeSetup, FirstTimeSetupState};
use crate::app::focus::{Focus, FocusProp, FocusProvider};
use crate::app::modeline::{Command, Mode, Modeline, ModelineState};
use crate::utils::{Cached, InsertBetweenExt};

pub mod component;
pub mod first_time_setup;
pub mod focus;
pub mod modeline;

#[must_use]
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
    // let exec = LocalExecutor::new();
    let exec = Executor::new();

    let mut emu = Emu::default();
    let log_rx = emu.tty.set_channeled();
    let cpu_state = Cached::new(emu.cpu.clone());
    let emu = Arc::new(RwLock::new(emu));
    let (config_tx, config_rx) = flume::unbounded();
    let (emu_cmd_tx, emu_cmd_rx) = flume::unbounded();
    let (emu_info_tx, emu_info_rx) = flume::unbounded();

    let emu_2 = emu.clone();
    exec.spawn(emu_task_v2(emu_2, config_rx, emu_cmd_rx, emu_info_tx))
        .detach();

    let future = exec.run(async {
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
            pipeline_report: EmuDynarecPipelineReport::not_started(),
            frequency: None,

            modeline_state: ModelineState { mode: Mode::Normal },
            first_time_setup_state: FirstTimeSetupState::default(),
            main_page_state: MainPageState {
                fetch_summary: None,
                fetch_summary_v2: None,
                emu_state: EmuState::Uninitialized.into(),
                cpu_inspector_state: CpuInspectorState {
                    emu: None,
                    cpu_state,
                    current_tab: CpuInspectorTab::Cpu,
                    cpu_tab_state: CpuTabState {
                        table_state: TableState::default(),
                        show_zero:   true,
                    },
                    cop0_tab_state: Cop0TabState {
                        table_state: TableState::default(),
                        show_zero:   true,
                    },
                },
                ops_list_state: OpsListState::default(),
                emu: None,
                emu_cmd_tx,
                action_state: ActionState {
                    chord: ActionChord::None,
                    add_breakpoint_popup: None,
                },
                memory_inspector_state: MemoryInspectorState {
                    emu: emu.clone(),
                    dims: None,
                    loaded_address: 0xbfc0_0000,
                    selected_address: 0xbfc0_0000,
                    loaded: Vec::new(),
                    loaded_address_tags: Vec::new(),
                    table_state: TableState::new(),
                    address_bar_state: MemoryInspectorAddressBarState {
                        input: Input::new(String::new()),
                        mode:  MemoryInspectorAddressBarMode::Normal,
                    },
                },
                tty_state: TtyState {
                    output: CircularBuffer::new(),
                    log_rx,
                },
            },
            pipeline_report_v2: PipelineV2Stage::Uninit,
        };

        let mut freq_updates_received = 0;
        loop {
            if app_state.exit {
                return Ok(());
            }
            terminal.draw(|frame| {
                let result: color_eyre::Result<()> = try {
                    let mut app = App(app_state.focus.props());
                    if let Ok(true) = event::poll(Duration::from_millis(0)) {
                        let event = event::read().map_err(|err| err.into())?;
                        app.handle_input(event, &mut app_state, frame.area())?;
                    }
                    for info in emu_info_rx.try_iter() {
                        match info {
                            EmuInfo::PipelineUpdateV2(stage) => {
                                app_state.pipeline_report_v2 = stage;
                            }
                            EmuInfo::PipelineUpdate(report) => {
                                app_state.pipeline_report = report;
                                app_state.main_page_state.memory_inspector_state.fetch();
                            }
                            EmuInfo::Ref(emu) => {
                                app_state.emu = Some(emu.clone());
                                app_state.main_page_state.emu = Some(emu.clone());
                                app_state.main_page_state.cpu_inspector_state.emu =
                                    Some(emu.clone());
                            }
                            EmuInfo::Fetch(fetch_summary) => {
                                app_state.main_page_state.fetch_summary = Some(fetch_summary);
                            }
                            EmuInfo::FetchV2(fetch_summary) => {
                                app_state.main_page_state.fetch_summary_v2 = Some(fetch_summary);
                            }
                            EmuInfo::StateUpdate(emu_state) => {
                                app_state.main_page_state.emu_state = emu_state.into();
                            }
                            EmuInfo::FrequencyUpdate(new) => match app_state.frequency {
                                Some(freq) => {
                                    freq_updates_received += 1usize;
                                    app_state.frequency =
                                        Some(freq + new / freq_updates_received as f64);
                                }
                                None => {
                                    app_state.frequency = Some(new);
                                    freq_updates_received = 1;
                                }
                            },
                        }
                    }
                    app_state.focus.process();
                    app.clone()
                        .render(frame.area(), frame.buffer_mut(), &mut app_state)?;
                };
                if let Err(err) = result {
                    app_state.error = Some(err);
                }
            })?;
            smol::Timer::after(Duration::from_secs_f64(1.0 / 60.0)).await;
        }
    });
    smol::block_on(future)
}

pub enum EmuCmd {
    StepJit,
    Run,
    Pause,
    AddBreakpoint(u32),
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
    PipelineUpdateV2(PipelineV2Stage),
    FrequencyUpdate(f64),
    StateUpdate(EmuState),
    Fetch(FetchSummary),
    FetchV2(Vec<FetchedOp>),
    Ref(Arc<RwLock<Emu>>),
}

async fn emu_task_v2(
    emu: Arc<RwLock<Emu>>,
    config_rx: Receiver<AppConfig>,
    emu_cmd_rx: Receiver<EmuCmd>,
    emu_info_tx: Sender<EmuInfo>,
) -> Result<()> {
    emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Uninitialized))?;

    emu_info_tx.send(EmuInfo::StateUpdate(EmuState::WaitingForConfig))?;
    let Ok(config) = config_rx.recv() else {
        emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Error(eyre!(
            "emu thread received config with no bios path"
        ))))?;
        return Err(eyre!("emu thread received config with no bios path"));
    };
    emu_info_tx.send(EmuInfo::StateUpdate(EmuState::SettingUp))?;
    {
        let mut emu = emu.get_mut();

        emu.set_bios_path(config.bios_path.expect("config must contain bios path"));
        match emu.load_bios() {
            Ok(()) => {}
            Err(err) => {
                emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Error(err.into())))?;
                return Ok(());
            }
        }
        emu.jump_to_bios();
    }
    emu_info_tx.send(EmuInfo::StateUpdate(EmuState::Running))?;
    emu_info_tx.send(EmuInfo::Ref(emu.clone()))?;

    let mut pipeline = {
        let emu = emu.get();
        PipelineV2::new(&emu)
    };

    let mut breakpoints = HashSet::<u32>::new();
    let mut running = false;
    let mut elapsed = Duration::new(0, 0);
    let mut step_jit = async |mut pipeline: PipelineV2| -> color_eyre::Result<PipelineV2> {
        let new_emu = emu.clone();
        let (pipeline, took) =
            smol::unblock(move || -> color_eyre::Result<(PipelineV2, Duration)> {
                let mut emu = new_emu.get_mut();
                let now = Instant::now();
                pipeline = pipeline.step(&mut emu)?;
                Ok((pipeline, now.elapsed()))
            })
            .await?;
        elapsed += took;
        let fetch = dynarec_v2::FETCH_CHANNEL.1.try_iter().collect::<Vec<_>>();
        if !fetch.is_empty() {
            emu_info_tx.send_async(EmuInfo::FetchV2(fetch)).await?;
        }
        if let PipelineV2::Called { times, .. } = &pipeline {
            let cycles_elapsed = emu.get().cpu.d_clock as f64 * *times as f64;
            if !elapsed.is_zero() {
                let frequency = cycles_elapsed / elapsed.as_secs_f64();
                elapsed = Duration::new(0, 0);
                emu_info_tx
                    .send_async(EmuInfo::FrequencyUpdate(frequency))
                    .await?;
            }
        }
        emu_info_tx
            .send_async(EmuInfo::PipelineUpdateV2(pipeline.stage()))
            .await?;
        Ok(pipeline)
    };

    loop {
        // if let Ok(new_config) = config_rx.try_recv() {
        //     config = new_config;
        // }
        for cmd in emu_cmd_rx.try_iter() {
            match cmd {
                EmuCmd::StepJit => {
                    pipeline = step_jit(pipeline).await?;
                }
                EmuCmd::Run => {
                    running = true;
                }
                EmuCmd::Pause => {
                    running = false;
                }
                EmuCmd::AddBreakpoint(addr) => {
                    breakpoints.insert(addr);
                }
            };
        }
        if running {
            pipeline = step_jit(pipeline).await?;
            if breakpoints.contains(&emu.get().cpu.pc) {
                running = false;
            }
        }
        smol::future::yield_now().await;
    }
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
    #[allow(dead_code)]
    error: Option<color_eyre::Report>,
    emu: Option<Arc<RwLock<Emu>>>,
    pipeline_report: EmuDynarecPipelineReport,
    pipeline_report_v2: PipelineV2Stage,
    frequency: Option<f64>,

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
        area: Rect,
    ) -> Result<Self::ComponentSummary> {
        let cmd = Modeline(self.0.clone().prop()).handle_input(
            event.clone(),
            &mut state.modeline_state,
            area,
        )?;
        match cmd {
            Command::Escape | Command::None => {}
            Command::Quit => state.exit = true,
        }

        match state.screen {
            Screen::FirstTimeSetup => {
                let action = FirstTimeSetup(self.0.prop()).handle_input(
                    event,
                    &mut state.first_time_setup_state,
                    area,
                )?;
                match action {
                    TypingAction::Submit(path) => {
                        state.app_config.bios_path = Some(path);
                        confy::store("pchan-debugger", "config", state.app_config.clone())?;
                        state.screen = Screen::Main;
                        self.0.prop::<OpsList>().push_focus();
                        state.config_tx.send(state.app_config.clone())?;
                    }
                    TypingAction::Escape | TypingAction::Pending | TypingAction::Enter => {}
                }
            }
            Screen::Main => {
                MainPage(self.0.prop(), state.pipeline_report_v2, state.frequency).handle_input(
                    event,
                    &mut state.main_page_state,
                    area,
                )?;
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
            Screen::Main => MainPage(self.0.prop(), state.pipeline_report_v2, state.frequency)
                .render(area, buf, &mut state.main_page_state)?,
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

pub struct MainPage(FocusProp<MainPage>, PipelineV2Stage, Option<f64>);
pub struct MainPageState {
    emu_state: Arc<EmuState>,
    emu: Option<Arc<RwLock<Emu>>>,
    emu_cmd_tx: Sender<EmuCmd>,
    fetch_summary: Option<FetchSummary>,
    fetch_summary_v2: Option<Vec<FetchedOp>>,
    ops_list_state: OpsListState,
    cpu_inspector_state: CpuInspectorState,
    action_state: ActionState,
    memory_inspector_state: MemoryInspectorState,
    tty_state: TtyState,
}

pub struct OpsList(Arc<EmuState>, FocusProp<OpsList>);
#[derive(Debug, Clone)]
pub struct OpsListData {
    ops:  Vec<Line<'static>>,
    hash: u64,
}
#[derive(Default)]
pub struct OpsListState {
    list_state: ListState,
    data:       Option<OpsListData>,
}

impl OpsListData {
    #[must_use]
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
                    Span::raw(format!("{}   {}", hex(*address), op)).style(
                        match (address.shr(2) % 2u32 == 0u32, op.is_block_boundary()) {
                            (
                                _,
                                Some(BoundaryType::Block { .. } | BoundaryType::BlockSplit { .. }),
                            ) => Style::new().red(),
                            (true, _) => Style::new().fg(Color::Rgb(250 - 15, 250 - 15, 250 - 15)),
                            (false, _) => Style::new(),
                        },
                    ),
                )
            })
            .insert_between(
                |a, b| a.0.abs_diff(*b.0).ge(&8),
                || (&0u32, Span::from("...").style(Style::new().dim())),
            )
            .map(|(_, line)| line.into())
            .collect::<Vec<Line>>();
        let mut hasher = DefaultHasher::new();
        decoded_ops.hash(&mut hasher);
        Self {
            ops:  decoded_ops,
            hash: hasher.finish(),
        }
    }
    pub fn from_fetch_v2(decoded_ops: &[FetchedOp]) -> Self {
        let decoded_ops = decoded_ops
            .iter()
            .map(|(address, op)| {
                (
                    address,
                    Span::raw(format!("{}   {}", hex(*address), op)).style(
                        match (address.shr(2) % 2u32 == 0u32, op.is_boundary()) {
                            (_, true) => Style::new().red(),
                            (true, _) => Style::new().fg(Color::Rgb(250 - 15, 250 - 15, 250 - 15)),
                            (false, _) => Style::new(),
                        },
                    ),
                )
            })
            .insert_between(
                |a, b| a.0.abs_diff(*b.0).ge(&8),
                || (&0u32, Span::from("...").style(Style::new().dim())),
            )
            .map(|(_, line)| line.into())
            .collect::<Vec<Line>>();
        let mut hasher = DefaultHasher::new();
        decoded_ops.hash(&mut hasher);
        Self {
            ops:  decoded_ops,
            hash: hasher.finish(),
        }
    }
}

impl Focus for OpsList {
    fn focus_prev() -> Option<focus::FocusId> {
        Some(MemoryInspector::as_focus())
    }
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
        _: Rect,
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
        let OpsList(emu_state, _) = self;
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

impl Component for MainPage {
    type ComponentState = MainPageState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let [left, mid, memory] = Layout::horizontal([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 4),
            Constraint::Min(32),
        ])
        .areas(area);
        let [ops, tty] = Layout::vertical([Constraint::Min(1), Constraint::Max(12)]).areas(left);
        let ops_focus = self.0.prop::<OpsList>();
        let ops_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1))
            .border_style(ops_focus.style())
            .title("OPコード");
        let ops_inner = ops_block.inner(ops);
        ops_block.render(ops, buf);
        let ops = ops_inner;

        match (&mut state.ops_list_state.data, &mut state.fetch_summary_v2) {
            (None, None) => {}
            (None, Some(fetch_summary)) => {
                state.ops_list_state.data = Some(OpsListData::from_fetch_v2(fetch_summary));
            }
            (Some(_), None) => {} // unreachable in practice (probably)
            (Some(old), Some(new)) => {
                let mut hasher = DefaultHasher::new();
                new.hash(&mut hasher);
                let new_hash = hasher.finish();
                if old.hash != new_hash {
                    state.ops_list_state.data = Some(OpsListData::from_fetch_v2(new));
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

        let tty_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title("tty");
        let tty_inner = tty_block.inner(tty);
        tty_block.render(tty, buf);
        Tty.render(tty_inner, buf, &mut state.tty_state)?;

        let actions_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::uniform(1))
            .title("Actions");
        let actions_block_inner = actions_block.inner(actions);
        actions_block.render(actions, buf);
        Actions(&state.emu_cmd_tx, self.1, self.0.prop(), self.2).render(
            actions_block_inner,
            buf,
            &mut state.action_state,
        )?;

        let memory_inspector_focus = self.0.prop();
        let memory_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(memory_inspector_focus.style())
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
        area: Rect,
    ) -> Result<Self::ComponentSummary> {
        let ops_focus = self.0.prop::<OpsList>();
        OpsList(state.emu_state.clone(), ops_focus).handle_input(
            event.clone(),
            &mut state.ops_list_state,
            area,
        )?;
        CpuInspector(self.0.prop()).handle_input(
            event.clone(),
            &mut state.cpu_inspector_state,
            area,
        )?;
        Actions(&state.emu_cmd_tx, self.1, self.0.prop(), self.2).handle_input(
            event.clone(),
            &mut state.action_state,
            area,
        )?;
        MemoryInspector(self.0.prop()).handle_input(
            event,
            &mut state.memory_inspector_state,
            area,
        )?;
        Ok(())
    }
}

pub struct Tty;
pub struct TtyState {
    output: CircularBuffer<256, Arc<str>>,
    log_rx: Receiver<Arc<str>>,
}

impl Component for Tty {
    type ComponentState = TtyState;
    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        _: event::Event,
        _: &mut Self::ComponentState,
        _: Rect,
    ) -> Result<Self::ComponentSummary> {
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        for line in state.log_rx.try_iter() {
            tracing::info!("{}", line.trim());
            state.output.push_back(line);
        }
        Widget::render(
            List::new(state.output.iter().map(|line| Line::from(&**line))),
            area,
            buf,
        );
        Ok(())
    }
}

pub struct CpuInspector(FocusProp<Self>);
pub struct CpuInspectorState {
    emu: Option<Arc<RwLock<Emu>>>,
    cpu_state: Cached<Cpu>,
    current_tab: CpuInspectorTab,
    cpu_tab_state: CpuTabState,
    cop0_tab_state: Cop0TabState,
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
        area: Rect,
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
        match state.current_tab {
            CpuInspectorTab::Cpu => {
                CpuTab(&state.cpu_state).handle_input(event, &mut state.cpu_tab_state, area)?;
            }
            CpuInspectorTab::Cop0 => {
                Cop0Tab(&state.cpu_state).handle_input(event, &mut state.cop0_tab_state, area)?
            }
            CpuInspectorTab::Gte => {}
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
        if let Ok(emu) = emu.try_read() {
            state.cpu_state.update(&emu.cpu);
        }

        let [summary_area, tabs_area] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(area);

        Paragraph::new(format!("PC: 0x{:08x}", state.cpu_state.pc)).render(summary_area, buf);

        let [tabs_area, tabs_inner] =
            Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(tabs_area);
        Tabs::new(CpuInspectorTab::iter().map(|el| el.to_string()))
            .style(Style::default().white())
            .highlight_style(Style::default().yellow())
            .select(state.current_tab as usize)
            .divider(symbols::border::PLAIN.vertical_left)
            .render(tabs_area, buf);
        match state.current_tab {
            CpuInspectorTab::Cpu => {
                CpuTab(&state.cpu_state).render(tabs_inner, buf, &mut state.cpu_tab_state)?
            }
            CpuInspectorTab::Cop0 => {
                Cop0Tab(&state.cpu_state).render(tabs_inner, buf, &mut state.cop0_tab_state)?
            }
            CpuInspectorTab::Gte => {}
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

pub struct CpuTab<'a>(&'a Cached<Cpu>);
pub struct CpuTabState {
    table_state: TableState,
    show_zero:   bool,
}

impl<'a> Component for CpuTab<'a> {
    type ComponentState = CpuTabState;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
        _: Rect,
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
        let rows = {
            self.0
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
                .collect::<Vec<_>>()
        };

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

pub struct Cop0Tab<'a>(&'a Cached<Cpu>);
pub struct Cop0TabState {
    table_state: TableState,
    show_zero:   bool,
}

impl<'a> Component for Cop0Tab<'a> {
    type ComponentState = Cop0TabState;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
        _: Rect,
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
        let [table, value] =
            Layout::vertical([Constraint::Min(2), Constraint::Length(3)]).areas(area);
        let rows = {
            self.0
                .cop0
                .reg
                .iter()
                .enumerate()
                .filter(|(_, value)| if !state.show_zero { **value != 0 } else { true })
                .map(|(reg, value)| {
                    Row::new([
                        Cow::Owned(format!("$cop0reg{}", reg)),
                        Cow::Borrowed(hex(*value)),
                    ])
                    .style(if *value == 0 {
                        Style::new().dim()
                    } else {
                        Style::new()
                    })
                })
                .collect::<Vec<_>>()
        };

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
            table,
            buf,
            &mut state.table_state,
        );

        Ok(())
    }
}

pub struct Actions<'a>(
    &'a Sender<EmuCmd>,
    // &'a EmuDynarecPipelineReport,
    PipelineV2Stage,
    FocusProp<ActionsAddBreakpoint>,
    Option<f64>,
);

pub struct ActionFocus;

impl Focus for ActionFocus {}

pub struct ActionState {
    chord: ActionChord,
    add_breakpoint_popup: Option<Input>,
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
        _: Rect,
    ) -> Result<Self::ComponentSummary> {
        if self.2.is_focused() {
            match (&mut state.add_breakpoint_popup, &event) {
                (Some(input), event::Event::Key(key!(Enter, Press))) => {
                    self.0
                        .send(EmuCmd::AddBreakpoint(parse_address(input.value())?))?;
                    self.2.unfocus();
                    state.add_breakpoint_popup = None;
                }
                (Some(_input), event::Event::Key(key!(Esc, Press))) => {
                    self.2.unfocus();
                    state.add_breakpoint_popup = None;
                }
                (Some(input), event) => {
                    input.handle_event(event);
                }
                _ => {}
            };
        } else {
            match (&state.chord, event) {
                (ActionChord::None, event::Event::Key(key!(Char('n'), Press))) => {
                    self.0.send(EmuCmd::StepJit)?;
                }
                (ActionChord::None, event::Event::Key(key!(Char('p'), Press))) => {
                    self.0.send(EmuCmd::Pause)?;
                }
                (ActionChord::None, event::Event::Key(key!(Char('r'), Press))) => {
                    self.0.send(EmuCmd::Run)?;
                }
                (ActionChord::None, event::Event::Key(key!(Char('b'), Press))) => {
                    state.chord = ActionChord::Breakpoint;
                }
                (ActionChord::Breakpoint, event::Event::Key(key!(Char('a'), Press))) => {
                    state.add_breakpoint_popup = Some(Input::new("".to_owned()));
                    self.2.push_focus();
                }
                (ActionChord::Breakpoint, event::Event::Key(key!(Char('d'), Press))) => {
                    todo!("delete breakpoint")
                }
                (ActionChord::Breakpoint, event::Event::Key(_)) => {
                    state.chord = ActionChord::None;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let [actions, report, stage_list] =
            Layout::vertical([Constraint::Max(6), Constraint::Max(3), Constraint::Max(6)])
                .areas(area);
        Paragraph::new(
            "<n>  next
            <r>  run 
            <ba> add breakpoint
            <bd> delete breakpoint
            "
            .to_string(),
        )
        .wrap(Wrap { trim: true })
        .render(actions, buf);

        let freq = match self.3.map(|freq| freq / 1_000_000.) {
            Some(freq) => Cow::Owned(format!("{freq:.2}MHz")),
            None => Cow::Borrowed("N/A"),
        };
        let [line_area, elapsed_area] =
            Layout::vertical([Constraint::Length(2), Constraint::Length(1)]).areas(report);
        LineGauge::default()
            .block(
                Block::new()
                    .title(format!("Emu Stage :: {} @ {}", self.1, freq))
                    .title_style(Style::new().bold()),
            )
            .unfilled_style(Style::new().dark_gray())
            .filled_style(Style::new().green().on_black())
            .line_set(symbols::line::THICK)
            .ratio(self.1 as u8 as f64 / { PipelineV2::COUNT.saturating_sub(1) as f64 })
            .render(line_area, buf);

        let progress = PipelineV2Stage::iter()
            .skip(1)
            .enumerate()
            .map(|(i, state)| (i, format!(" - {}", state)))
            .map(|(i, state)| {
                Line::from(state).style(if i as u8 <= (self.1 as u8).saturating_sub(1) {
                    Style::new().white()
                } else {
                    Style::new().dark_gray()
                })
            });
        Text::from_iter(progress).render(stage_list, buf);

        // TODO: break out into separate component

        if let Some(input) = &state.add_breakpoint_popup {
            let [area] = Layout::vertical([Constraint::Length(3)]).areas(area);
            Clear.render(area, buf);
            Paragraph::new(input.value())
                .block(
                    Block::bordered()
                        .border_style(self.2.style())
                        .border_type(BorderType::Rounded)
                        .title("Add breakpoint"),
                )
                .render(area, buf);
        }

        Ok(())
    }
}

pub struct ActionsAddBreakpoint;

impl Focus for ActionsAddBreakpoint {}

pub struct MemoryInspector(FocusProp<Self>);
pub struct MemoryInspectorState {
    emu: Arc<RwLock<Emu>>,
    loaded_address: u32,
    selected_address: u32,
    dims: Option<(u16, u16)>,
    loaded: Vec<const_hex::Buffer<1>>,
    loaded_address_tags: Vec<const_hex::Buffer<4, true>>,
    table_state: TableState,
    address_bar_state: MemoryInspectorAddressBarState,
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

impl MemoryInspectorState {
    fn fetch(&mut self) {
        let Some((rows, cols)) = self.dims else {
            return;
        };

        let Ok(emu) = &self.emu.try_read() else {
            return;
        };

        self.loaded.clear();
        (self.loaded_address..)
            .take((rows * MemoryInspector::COLUMNS) as usize)
            .map(|address| emu.try_read::<u8>(address))
            .map(|byte| byte.unwrap_or(0x0))
            .map(|byte| const_hex::const_encode::<_, false>(&[byte]))
            .collect_into(&mut self.loaded);

        self.loaded_address_tags.clear();
        self.loaded
            .chunks(cols as usize)
            .enumerate()
            .map(|(row, _)| self.loaded_address + row as u32 * cols as u32)
            .map(|address| const_hex::const_encode::<_, true>(&address.to_be_bytes()))
            .collect_into(&mut self.loaded_address_tags);
    }
    fn update_address(&mut self, addr: u32) {
        let Some((rows, _)) = self.dims else {
            return;
        };
        let cols = MemoryInspector::COLUMNS as u32;
        let page = self.loaded_address..(self.loaded_address.saturating_add(rows as u32 * cols));
        if !page.contains(&addr) {
            self.loaded_address = addr.next_multiple_of(cols) - cols;
            // pagination
            let row_idx = addr / MemoryInspector::COLUMNS as u32;
            self.loaded_address =
                (row_idx / rows as u32) * rows as u32 * MemoryInspector::COLUMNS as u32;
        }
        self.selected_address = addr;

        self.fetch();
    }
    fn update_selected(&mut self) {
        let selected = self
            .loaded
            .chunks(MemoryInspector::COLUMNS as usize)
            .enumerate()
            .find_map(|(row_idx, row)| {
                row.iter().enumerate().find_map(|(col_idx, _)| {
                    let address = self.loaded_address as usize
                        + (row_idx * MemoryInspector::COLUMNS as usize + col_idx);
                    if address == self.selected_address as usize {
                        Some((row_idx, col_idx))
                    } else {
                        None
                    }
                })
            });
        self.table_state.select_cell(selected);
    }
}

impl MemoryInspector {
    const COLUMNS: u16 = 16;

    fn render_hex_table(
        &self,
        area: Rect,
        buf: &mut Buffer,
        state: &mut MemoryInspectorState,
    ) -> Result<()> {
        const COL_WIDTH: u16 = 3;

        let [tags, area] =
            Layout::horizontal([Constraint::Max(12), Constraint::Min(4)]).areas(area);
        // area.width = area.width.max(50);
        let cols = Self::COLUMNS;
        let rows = area.height - 2;
        state.dims = Some((rows, cols));

        if cols <= 1 {
            "Window Too small".render(area, buf);
            return Ok(());
        }

        if state.loaded.is_empty() {
            state.fetch();
        }

        let rows = {
            state
                .loaded
                .chunks(cols as usize)
                .enumerate()
                .map(|(row_idx, row)| {
                    row.iter().enumerate().map(move |(col_idx, cell)| {
                        Span::raw(cell.as_str()).style(if cell.as_str() == "00" {
                            Style::new().dark_gray()
                        } else if (row_idx + col_idx) % 2 == 0 {
                            Style::new().white()
                        } else {
                            Style::new().fg(Color::Rgb(255 - 15, 255 - 15, 255 - 15))
                        })
                    })
                })
                .map(Row::new)
                .collect::<Vec<_>>()
        };

        if state.table_state.selected_cell().is_none() {
            state.table_state.select_cell(Some((0, 0)));
        }

        StatefulWidget::render(
            Table::new(
                rows,
                [
                    Constraint::Length(COL_WIDTH - 1),
                    Constraint::Length(COL_WIDTH - 1),
                    Constraint::Length(COL_WIDTH - 1),
                    Constraint::Length(COL_WIDTH),
                ]
                .repeat(4),
            )
            .column_spacing(0)
            .style(Style::new().gray())
            .cell_highlight_style(Style::new().on_gray().black())
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::new().white())
                    .title("Hex"),
            ),
            area,
            buf,
            &mut state.table_state,
        );

        StatefulWidget::render(
            List::new(state.loaded_address_tags.iter().map(|row| row.as_str()))
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .title("Address"),
                )
                .highlight_style(Style::new().bg(Color::Gray).fg(Color::Black)),
            tags,
            buf,
            &mut ListState::default().with_selected(state.table_state.selected()),
        );

        Ok(())
    }
}

impl Component for MemoryInspector {
    type ComponentState = MemoryInspectorState;

    type ComponentSummary = ();

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
        area: Rect,
    ) -> Result<Self::ComponentSummary> {
        let new_selected = MemoryInspectorAddressBar {
            focus: self.0.prop(),
            selected_address: state.selected_address,
        }
        .handle_input(event.clone(), &mut state.address_bar_state, area);
        match new_selected {
            Ok(Some(address)) => {
                state.update_address(address);
            }
            Ok(_) => {}
            Err(_) => {
                state.address_bar_state.normal(self.0.prop());
            }
        }

        if let Some((_, cols)) = state.dims {
            match event {
                event::Event::Key(key!(Char('l'), Press, KeyModifiers::CONTROL)) => {
                    state.update_address(state.selected_address + 1);
                    self.0.focus();
                }
                event::Event::Key(key!(Char('h'), Press, KeyModifiers::CONTROL)) => {
                    state.update_address(state.selected_address - 1);
                    self.0.focus();
                }
                event::Event::Key(key!(Char('j'), Press, KeyModifiers::CONTROL)) => {
                    state.update_address(state.selected_address + cols as u32);
                    self.0.focus();
                }
                event::Event::Key(key!(Char('k'), Press, KeyModifiers::CONTROL)) => {
                    state.update_address(state.selected_address - cols as u32);
                    self.0.focus();
                }
                _ => {}
            }
            state.update_selected();
        }

        if !self.0.is_focused() {
            return Ok(());
        }

        match event {
            event::Event::Key(key!(Char('g'), Press)) => {
                state.address_bar_state.go(self.0.prop());
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
        Ok(())
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        let [table, address_area] =
            Layout::vertical([Constraint::Min(2), Constraint::Max(3)]).areas(area);

        self.render_hex_table(table, buf, state)?;
        MemoryInspectorAddressBar {
            focus: self.0.prop(),
            selected_address: state.selected_address,
        }
        .render(address_area, buf, &mut state.address_bar_state)?;

        Ok(())
    }
}

pub struct MemoryInspectorAddressBar {
    focus: FocusProp<Self>,
    selected_address: u32,
}
pub struct MemoryInspectorAddressBarState {
    input: Input,
    mode:  MemoryInspectorAddressBarMode,
}

pub enum MemoryInspectorAddressBarMode {
    Normal,
    Go,
}

impl Component for MemoryInspectorAddressBar {
    type ComponentState = MemoryInspectorAddressBarState;

    type ComponentSummary = Option<u32>;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
        _: Rect,
    ) -> Result<Self::ComponentSummary> {
        if !self.focus.is_focused() {
            return Ok(None);
        }
        state.input.handle_event(&event);

        use MemoryInspectorAddressBarMode as Mode;
        match (&state.mode, &event) {
            (Mode::Go, event::Event::Key(key!(Esc, Press))) => {
                state.mode = Mode::Normal;
                state.input.reset();
                state.normal(self.focus.prop());

                Ok(None)
            }
            (Mode::Go, event::Event::Key(key!(Enter, Press))) => {
                let address = match state.input.value().split_at_checked(2) {
                    Some(("0x" | "0X", address)) => address,
                    Some(_) => state.input.value(),
                    None => state.input.value(),
                };
                let address = u32::from_str_radix(address, 16)?;

                state.mode = Mode::Normal;
                state.input.reset();
                state.normal(self.focus.prop());

                Ok(Some(address))
            }
            _ => Ok(None),
        }
    }

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
        use MemoryInspectorAddressBarMode as Mode;
        match state.mode {
            Mode::Normal => {
                Paragraph::new(format!("0x{:08x}", self.selected_address))
                    .block(
                        Block::bordered()
                            .border_style(self.focus.style())
                            .border_type(BorderType::Rounded)
                            .title("Address"),
                    )
                    .render(area, buf);
            }
            Mode::Go => {
                Paragraph::new(state.input.value())
                    .block(
                        Block::bordered()
                            .border_style(self.focus.style())
                            .border_type(BorderType::Rounded)
                            .title("Go to"),
                    )
                    .render(area, buf);
            }
        }

        Ok(())
    }
}

impl MemoryInspectorAddressBarState {
    fn go(&mut self, focus: FocusProp<MemoryInspectorAddressBar>) {
        focus.push_focus();
        self.mode = MemoryInspectorAddressBarMode::Go;
        self.input.reset();
    }
    fn normal(&mut self, focus: FocusProp<MemoryInspectorAddressBar>) {
        focus.unfocus();
        self.mode = MemoryInspectorAddressBarMode::Normal;
        self.input.reset();
    }
}

impl Focus for MemoryInspectorAddressBar {}

fn parse_address(str: &str) -> color_eyre::Result<u32> {
    let address = match str.split_at_checked(2) {
        Some(("0x" | "0X", address)) => address,
        Some(_) => str,
        None => str,
    };
    let address = u32::from_str_radix(address, 16)?;
    Ok(address)
}

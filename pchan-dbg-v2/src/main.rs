#![allow(clippy::collapsible_if)]
#![feature(iter_array_chunks)]
#![feature(duration_millis_float)]

pub mod display;
pub mod init;
#[path = "./lipgloss-colors.rs"]
pub mod lipgloss_colors;
#[path = "./widgets/widgets.rs"]
pub mod widgets;

use std::{
    collections::VecDeque,
    io::stdout,
    ops::RangeInclusive,
    sync::Arc,
    time::{Duration, Instant},
};

use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture};
use crossterm_simple_event::CrosstermSimpleEvent;
use edtui::{
    EditorEventHandler, EditorMode, EditorState, EditorStatusLine, EditorTheme, EditorView,
};
use image::{DynamicImage, RgbaImage};
use miette::{Context, IntoDiagnostic, Result, miette};
use pchan_emu::{
    Emu,
    bootloader::Bootloader,
    cpu::reg_str,
    debug::{Breakpoint, BreakpointKind},
    dynarec_v2::{
        Dynarec,
        emitters::{DecodedOp, DynarecOp},
        run_step,
    },
    io::{IO, vblank::VBlank},
};
use pchan_gpu::Renderer;
use pchan_utils::{hex, hex_pref};
use rat_imaginary::{ImageState, ImageWidget};
use ratatui::{
    DefaultTerminal, Frame, crossterm,
    layout::{Constraint, Layout, Margin, Rect},
    style::{Color, Style, Styled, Stylize},
    widgets::{Block, BorderType, Borders, Clear, List, ListState, Row, Table, TableState, Widget},
};
use wgpu::Extent3d;

use crate::{
    display::{DisplayState, draw_display},
    init::EnvVars,
    lipgloss_colors::LIPGLOSS,
    widgets::{
        button::{Button, ButtonResponse, ButtonState, ButtonStyles},
        checkbox::{Checkbox, CheckboxState},
    },
};

fn main() -> Result<()> {
    // miette_panic::install(miette_panic::PanicHookArgs::default());
    pchan_utils::init_tracing()
        .file(true)
        .panic_hook(false)
        .stdout(false)
        .call();
    let env = EnvVars::new()?;
    smol::block_on(run_app(&env))
}

struct AppState {
    emu: Emu,
    gpu: Arc<Renderer>,
}

struct Theme {
    fg:      Color,
    bg:      Color,
    primary: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            fg:      Color::from_u32(0xffffff),
            bg:      Color::Rgb(0, 0, 0),
            primary: LIPGLOSS[0][0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopMode {
    Poll,
    Event,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focused {
    Preview,
    Registers,
    Mips,
    Mem,
    Breakpoints,
}

fn reinit_emu(emu: &mut Emu) -> Result<()> {
    let audio = emu.spu.take_prod();
    let gpu = emu.gpu.conn.clone();
    let dbg = std::mem::take(&mut emu.dbg);

    *emu = Emu::default();
    emu.spu.put_prod(audio);
    emu.gpu.conn = gpu;
    emu.dbg = dbg;

    emu.load_bios().into_diagnostic()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();
    Ok(())
}

impl AppState {
    pub fn reinit(&mut self) {
        reinit_emu(&mut self.emu).unwrap();
    }
}

async fn run_app(env: &EnvVars) -> Result<()> {
    let mut emu = Emu::default();
    let mut gpu = Renderer::new().await;
    let mut audio = pchan_audio::AudioTask::new()?;
    {
        let mut dp_uniforms = gpu.display_uniforms.lock().unwrap();
        dp_uniforms.dp_res.x = 640;
        dp_uniforms.dp_res.y = 480;
        dp_uniforms.screen_rect.x = 640;
        dp_uniforms.screen_rect.y = 480;
    }
    let mut dp = DisplayState::new(&gpu);
    emu.set_bios_path(&env.bios_path);
    emu.load_bios().into_diagnostic()?;
    emu.cpu.jump_to_bios();
    emu.tty.set_tracing();
    gpu.connect_emu(&mut emu);
    pchan_bind::bind_audio(&mut audio, &mut emu);
    let stream = audio.start()?;
    std::mem::forget(stream);

    let mut state = AppState {
        emu,
        gpu: gpu.into(),
    };
    let mut tui_state = TuiState::new();
    tui_state.reg_list.select_first();
    state.gpu.clone().start();

    run(|term| {
        // let pipe = PipelineV2::new(&state.emu);
        let mut frame_time_sample_time = Duration::ZERO;
        let mut frame_time_samples = VecDeque::with_capacity(32);
        let mut frame_time_sum = 0u128;
        let mut dynarec = Box::new(Dynarec::default());
        _ = term.draw(|frame| {
            draw_app(frame, &mut tui_state, &state);
        });
        loop {
            if tui_state.quit {
                break;
            }

            if tui_state.emu_running {
                dynarec = run_step(&mut state.emu, dynarec);
                tui_state.mips_cursor = state.emu.cpu.pc;
                tui_state.exec_history.push_back(state.emu.cpu.pc);
                if tui_state.exec_history.len() > 100 {
                    tui_state.exec_history.pop_front();
                }
                if state.emu.dbg.stopped_on.is_some() {
                    tui_state.emu_running = false;
                    tui_state.loop_mode = LoopMode::Event;
                    state.emu.dbg.stopped_on = None;
                    term.draw(|frame| {
                        draw_app(frame, &mut tui_state, &state);
                    })
                    .unwrap();
                }
            }

            // pipe = pipe.run_once(&mut state.emu).unwrap();
            if state.emu.consume_vblank_signal() {
                let last_vblank = state.emu.gpu.last_vblank;
                let recorded_frame_time = last_vblank.elapsed();
                let img = draw_display(&state, &mut dp);
                let size = dp.output_tex.size();

                if tui_state.loop_mode == LoopMode::Poll {
                    tui_state.update_current_frame(Some((size, img)));
                    term.draw(|frame| {
                        draw_app(frame, &mut tui_state, &state);
                    })
                    .unwrap();
                }

                let effective_frame_time = last_vblank.elapsed();
                frame_time_samples.push_back(recorded_frame_time);
                frame_time_sum += recorded_frame_time.as_nanos();
                if frame_time_samples.len() >= 32 {
                    if let Some(sample) = frame_time_samples.pop_front() {
                        frame_time_sum = frame_time_sum.saturating_sub(sample.as_nanos());
                    }
                }
                frame_time_sample_time += effective_frame_time;
                if frame_time_sample_time >= Duration::from_secs(1) {
                    frame_time_sample_time -= Duration::from_secs(1);
                    let average_frame_time = frame_time_sum / frame_time_samples.len() as u128;
                    tui_state.frame_time = Duration::from_nanos_u128(average_frame_time);
                }
                let sleep_for = Duration::from_millis(16).saturating_sub(effective_frame_time);
                let spin_for = Duration::from_millis(3);

                if sleep_for > spin_for {
                    let sleep_for = sleep_for.saturating_sub(spin_for);
                    if let Ok(true) = crossterm::event::poll(sleep_for) {
                        let ev = crossterm::event::read().unwrap();
                        handle_event(&mut state, &mut tui_state, &ev);
                        term.draw(|frame| {
                            draw_app(frame, &mut tui_state, &state);
                        })
                        .unwrap();
                    }
                }
                let now = Instant::now();
                while now.elapsed() < spin_for {
                    if let Ok(true) = crossterm::event::poll(Duration::ZERO) {
                        let ev = crossterm::event::read().unwrap();
                        handle_event(&mut state, &mut tui_state, &ev);
                        term.draw(|frame| {
                            draw_app(frame, &mut tui_state, &state);
                        })
                        .unwrap();
                    }
                }

                state.emu.gpu.last_vblank = Instant::now();
            }

            match tui_state.loop_mode {
                LoopMode::Poll => {}
                LoopMode::Event => {
                    if let Ok(ev) = crossterm::event::read() {
                        handle_event(&mut state, &mut tui_state, &ev);
                        term.draw(|frame| {
                            draw_app(frame, &mut tui_state, &state);
                        })
                        .unwrap();
                    }
                }
            }
        }
        tui_state.framebuffer.clear_backbuffer();
        tui_state.framebuffer.swap_buffers();
        tui_state.framebuffer.clear_backbuffer();
        tui_state.framebuffer.swap_buffers();
    })
}

fn run(callback: impl FnOnce(&mut DefaultTerminal)) -> Result<()> {
    let mut term = ratatui::init();
    _ = crossterm::execute!(stdout(), EnableMouseCapture);
    let callback = std::panic::AssertUnwindSafe(callback);
    let result = std::panic::catch_unwind(move || {
        let callback = callback;
        let callback = callback.0;
        callback(&mut term);
    })
    .map_err(|err| {
        let err = format!("{:?}", err.downcast::<Box<dyn std::fmt::Debug>>());
        miette!(err)
    });
    ratatui::restore();
    _ = crossterm::execute!(stdout(), DisableMouseCapture);
    result
}

impl Drop for TuiState {
    fn drop(&mut self) {
        self.framebuffer.clear_backbuffer();
        self.framebuffer.swap_buffers();
        self.framebuffer.clear_backbuffer();
    }
}

impl TuiState {
    fn update_current_frame(&mut self, img: Option<(Extent3d, Vec<u8>)>) {
        if let Some((size, img)) = img {
            let img =
                RgbaImage::from_vec(size.width, size.height, img).map(DynamicImage::ImageRgba8);
            self.framebuffer.clear_frontbuffer();
            self.framebuffer.write_image(img.clone().unwrap()).unwrap();
            self.current_frame = img;
        }
    }
}

fn handle_event(state: &mut AppState, tui_state: &mut TuiState, ev: &event::Event) {
    match ev.simple().as_str() {
        "ctrl+c" | "q" => tui_state.quit = true,
        event => {
            if let ButtonResponse::Clicked = tui_state.mips_jump_to_pc_button.handle_event(ev) {
                tui_state.mips_cursor = state.emu.cpu.pc;
            };
            match (tui_state.focused, event) {
                (Focused::Preview, "tab" | "j") => tui_state.focused = Focused::Registers,
                (Focused::Preview, "l") => tui_state.focused = Focused::Mips,
                (Focused::Preview, "f") => tui_state.fullscreen = !tui_state.fullscreen,
                (Focused::Preview, " ") => {
                    tui_state.emu_running = !tui_state.emu_running;
                    match tui_state.loop_mode {
                        LoopMode::Poll => tui_state.loop_mode = LoopMode::Event,
                        LoopMode::Event => tui_state.loop_mode = LoopMode::Poll,
                    }
                }
                (Focused::Preview, "r") => {
                    state.reinit();
                }
                (Focused::Registers, "backtab" | "k") => tui_state.focused = Focused::Preview,
                (Focused::Registers, "l") => tui_state.focused = Focused::Mips,
                (Focused::Registers, "ctrl+k" | "up") => tui_state.reg_list.select_previous(),
                (Focused::Registers, "ctrl+j" | "down") => tui_state.reg_list.select_next(),
                (Focused::Registers, "g") => tui_state.reg_list.select_first(),
                (Focused::Registers, "shift+g") => tui_state.reg_list.select_last(),
                (Focused::Mips, "h") => tui_state.focused = Focused::Registers,
                (Focused::Mips, "ctrl+j") => {
                    tui_state.mips_cursor = tui_state.mips_cursor.saturating_add(4);
                }
                (Focused::Mips, "ctrl+k") => {
                    tui_state.mips_cursor = tui_state.mips_cursor.saturating_sub(4);
                }
                (Focused::Mips, "l" | "tab") => tui_state.focused = Focused::Mem,
                (Focused::Mem, mem_key) => match tui_state.jump_to_mem_address_pane.open {
                    false => match mem_key {
                        "ctrl+h" => tui_state.mem_cursor = tui_state.mem_cursor.saturating_sub(1),
                        "ctrl+j" => tui_state.mem_cursor = tui_state.mem_cursor.saturating_add(16),
                        "ctrl+k" => tui_state.mem_cursor = tui_state.mem_cursor.saturating_sub(16),
                        "ctrl+l" => tui_state.mem_cursor = tui_state.mem_cursor.saturating_add(1),
                        "g" => {
                            tui_state.jump_to_mem_address_pane.open = true;
                        }
                        "h" | "backtab" => tui_state.focused = Focused::Mips,
                        "j" | "tab" => {
                            tui_state.focused = Focused::Breakpoints;
                        }
                        _ => {}
                    },
                    true => match mem_key {
                        "enter" => {
                            let address = parse_hex_address(
                                &tui_state
                                    .jump_to_mem_address_pane
                                    .input
                                    .lines
                                    .iter_row()
                                    .flatten()
                                    .collect::<String>(),
                            );
                            // TODO: report error
                            if let Ok(address) = address {
                                tui_state.mem_cursor = address;
                                tui_state.mem_range = address..=address;
                                tui_state.jump_to_mem_address_pane.open = false;
                            }
                        }
                        _ => {
                            if mem_key == "esc"
                                && tui_state.jump_to_mem_address_pane.input.mode
                                    == EditorMode::Normal
                            {
                                tui_state.jump_to_mem_address_pane.open = false;
                            }
                            EditorEventHandler::vim_mode().on_event(
                                ev.clone(),
                                &mut tui_state.jump_to_mem_address_pane.input,
                            );
                        }
                    },
                },
                (Focused::Breakpoints, key) => match tui_state.add_breakpoint_pane.open {
                    false => match key {
                        "k" | "backtab" => tui_state.focused = Focused::Mem,
                        "h" => tui_state.focused = Focused::Mips,
                        "a" => {
                            tui_state.add_breakpoint_pane.open = true;
                            tui_state.add_breakpoint_pane.focus = Some(0);
                        }
                        "d" if let Some(selected) = tui_state.breakpoints_table.selected() => {
                            let key = state
                                .emu
                                .dbg
                                .breakpoints
                                .iter()
                                .enumerate()
                                .find_map(|(idx, (k, _))| (idx == selected).then_some(*k));
                            if let Some(key) = key {
                                state.emu.dbg.breakpoints.remove(&key);
                            }
                        }
                        "ctrl+j" => tui_state.breakpoints_table.select_next(),
                        "ctrl+k" => tui_state.breakpoints_table.select_previous(),
                        "g" => tui_state.breakpoints_table.select_first(),
                        "shift+g" => tui_state.breakpoints_table.select_last(),
                        _ => {}
                    },
                    true => match (tui_state.add_breakpoint_pane.focus, key) {
                        (Some(_), "esc") => {
                            tui_state.add_breakpoint_pane.focus = None;
                        }
                        (None, "esc") => {
                            tui_state.add_breakpoint_pane.open = false;
                        }
                        (None, _) => {
                            tui_state.add_breakpoint_pane.focus = Some(0);
                        }
                        (Some(0), "enter" | "tab" | "ctrl+j") => {
                            tui_state.add_breakpoint_pane.focus = Some(1)
                        }
                        (Some(0), _) => {
                            tui_state
                                .add_breakpoint_pane
                                .address_input_handler
                                .on_event(
                                    ev.clone(),
                                    &mut tui_state.add_breakpoint_pane.address_input,
                                );

                            if tui_state.add_breakpoint_pane.address_input.lines.is_empty() {
                                tui_state.add_breakpoint_pane.error = None;
                            }
                        }
                        (Some(4), "enter") => {
                            let r = tui_state.add_breakpoint_pane.checkboxes[0].value();
                            let w = tui_state.add_breakpoint_pane.checkboxes[1].value();
                            let x = tui_state.add_breakpoint_pane.checkboxes[2].value();
                            let mut kind = BreakpointKind::NONE;
                            if r {
                                kind |= BreakpointKind::READ;
                            }
                            if w {
                                kind |= BreakpointKind::WRITE;
                            }
                            if x {
                                kind |= BreakpointKind::EXECUTE;
                            }
                            let line = &tui_state.add_breakpoint_pane.address_input.lines;
                            let line = line.iter_row().flatten().collect::<String>();
                            match create_breakpoint(&line, kind) {
                                Ok(breakpoint) => {
                                    tui_state.add_breakpoint_pane.error = None;
                                    tui_state.add_breakpoint_pane.open = false;
                                    state
                                        .emu
                                        .dbg
                                        .breakpoints
                                        .insert(breakpoint.address, breakpoint);
                                }
                                Err(err) => {
                                    tui_state.add_breakpoint_pane.error = Some(err);
                                }
                            }
                        }
                        (Some(ref mut idx @ 1..=4), "j" | "ctrl+j") => {
                            *idx += 1;
                            *idx %= 5;
                            tui_state.add_breakpoint_pane.focus = Some(*idx);
                        }
                        (Some(ref mut idx @ 1..=4), "k" | "ctrl+k") => {
                            *idx -= 1;
                            tui_state.add_breakpoint_pane.focus = Some(*idx);
                        }
                        (Some(ref mut idx @ 1..=3), _) => {
                            match tui_state.add_breakpoint_pane.checkboxes[*idx - 1]
                                .handle_event(ev)
                            {
                                widgets::EventResponse::Next => {
                                    *idx += 1;
                                    *idx %= 5;
                                }
                                widgets::EventResponse::None => {}
                                widgets::EventResponse::GrabFocus => {}
                            }
                            tui_state.add_breakpoint_pane.focus = Some(*idx);
                        }
                        _ => {}
                    },
                },

                _ => {}
            }
        }
    }
}

fn parse_hex_address(address: &str) -> Result<u32> {
    let address = address.trim_start().trim_start_matches("0x").trim_end();
    let address: u32 = u32::from_str_radix(address, 16)
        .into_diagnostic()
        .wrap_err("invalid breakpoint")?;
    Ok(address)
}

fn create_breakpoint(address: &str, kind: BreakpointKind) -> Result<Breakpoint> {
    let address = parse_hex_address(address)?;
    Ok(Breakpoint {
        address,
        kind,
        enabled: true,
    })
}

struct TuiState {
    theme:         Theme,
    loop_mode:     LoopMode,
    emu_running:   bool,
    current_frame: Option<DynamicImage>,
    framebuffer:   ImageState,
    quit:          bool,
    fullscreen:    bool,
    frame_time:    Duration,
    reg_list:      TableState,
    focused:       Focused,
    mips_cursor:   u32,
    mips_range:    RangeInclusive<u32>,
    mem_cursor:    u32,
    mem_range:     RangeInclusive<u32>,
    exec_history:  VecDeque<u32>,

    mips_jump_to_pc_button:   ButtonState,
    add_breakpoint_pane:      AddBreakpointPane,
    jump_to_mem_address_pane: JumpToMemAddressPane,
    breakpoints_table:        TableState,
}

#[derive(Default)]
struct AddBreakpointPane {
    open:  bool,
    focus: Option<usize>,
    error: Option<miette::Report>,

    address_input:         EditorState,
    address_input_handler: EditorEventHandler,
    checkboxes:            [CheckboxState; 3],
}

#[derive(Default)]
struct JumpToMemAddressPane {
    open:  bool,
    input: EditorState,
}

impl TuiState {
    fn new() -> Self {
        TuiState {
            loop_mode:     LoopMode::Event,
            current_frame: None,
            framebuffer:   ImageState::new(),
            quit:          false,
            fullscreen:    false,
            frame_time:    Duration::ZERO,
            reg_list:      TableState::new(),
            theme:         Theme::default(),
            focused:       Focused::Preview,
            mips_cursor:   0x0,
            emu_running:   false,
            mips_range:    0x0..=0x0,
            mem_cursor:    0xbfc0_0000,
            mem_range:     0xbfc0_0000..=0xbfc0_0000,
            exec_history:  VecDeque::new(),

            mips_jump_to_pc_button:   ButtonState::new(),
            add_breakpoint_pane:      AddBreakpointPane::default(),
            breakpoints_table:        TableState::default().with_selected(Some(0)),
            jump_to_mem_address_pane: JumpToMemAddressPane::default(),
        }
    }

    fn focus_style(&self, value: Focused) -> Style {
        if self.focused == value {
            Style::new().fg(self.theme.primary)
        } else {
            Style::new()
        }
    }
}

fn draw_app(frame: &mut Frame, tui_state: &mut TuiState, state: &AppState) {
    let main_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title("+ 🐷🎗️ P-ちゃん dbg (v2) +".set_style(tui_state.theme.primary));
    let area = main_block.inner(frame.area());
    main_block.render(frame.area(), frame.buffer_mut());

    let [h1, h2, h3] = Layout::horizontal([
        Constraint::Ratio(1, 3),
        Constraint::Ratio(1, 3),
        Constraint::Ratio(1, 3),
    ])
    .areas(area);

    let (width, height) = match &tui_state.current_frame {
        Some(img) => (img.width(), img.height()),
        None => (640, 480),
    };
    let ar = height * 100 / width;
    let h1_w = h1.width;
    let img_h = h1_w * ar as u16 / 100;
    let img_h = (img_h * 2) / 3;
    let [img_area, rest] =
        Layout::vertical([Constraint::Length(img_h), Constraint::Fill(1)]).areas(h1);
    let frame_time = tui_state.frame_time.as_millis_f32();
    let fps = 1000.0 / frame_time;

    let img_block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title_bottom(format!(
            "+ {}x{} {:01.2}ms {:.0}fps +",
            width, height, frame_time, fps
        ))
        .title_top(" 📺 ")
        .title_top(" <spc> run • <f> fullscreen ".dim())
        .theme(&tui_state.theme)
        .focus_style(tui_state, Focused::Preview);
    let inner_img_area = img_block.inner(img_area);
    img_block.render(img_area, frame.buffer_mut());

    if tui_state.current_frame.is_some() {
        ImageWidget.render(inner_img_area, frame, &mut tui_state.framebuffer);
    } else {
        let area = inner_img_area.centered(Constraint::Length(10), Constraint::Length(1));
        frame.render_widget("何もない".set_style(Style::new().dim()), area);
    }

    draw_register_viewer(rest, frame, tui_state, state);
    draw_mips_assembly(h2, frame, tui_state, state);

    {
        let [mem_area, breakpoints_area] =
            Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(h3);
        draw_mem(mem_area, frame, tui_state, state);
        draw_breakpoints(breakpoints_area, frame, tui_state, state);
    }
}

fn draw_register_viewer(area: Rect, frame: &mut Frame, tui_state: &mut TuiState, state: &AppState) {
    if tui_state.focused != Focused::Registers {
        tui_state.reg_list.select(None);
    } else if tui_state.reg_list.selected().is_none() {
        tui_state.reg_list.select_first();
    }

    let area = {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title_top("+ registers +")
            .title_bottom(" <C-j>/<C-k> down/up ".dim())
            .theme(&tui_state.theme)
            .focus_style(tui_state, Focused::Registers);
        let a = block.inner(area);
        block.render(area, frame.buffer_mut());
        a
    };

    let pc_hex = hex(state.emu.cpu.pc);
    let rows = [Row::new(["pc", pc_hex.as_str()])];
    let regs_hex = state.emu.cpu.gpr.map(hex);
    let rows = rows.into_iter().chain((0..32).map(|r| {
        let mut style = if state.emu.cpu.gpr[r] == 0 {
            Style::from(tui_state.theme.fg).dim()
        } else {
            tui_state.theme.fg.into()
        };
        if r % 2 == 0 {
            style.fg = style.fg.darken(0.1);
        }
        Row::new([reg_str(r as u8), regs_hex[r].as_str()]).style(style)
    }));
    let table = Table::new(rows, [Constraint::Length(3), Constraint::Length(10)])
        .row_highlight_style(Style::new().bg(tui_state.theme.primary).bold().italic())
        .theme(&tui_state.theme);
    frame.render_stateful_widget(table, area, &mut tui_state.reg_list);
}

/// returns the offset
fn compute_infinite_list(
    cursor: &mut u32,
    page: &mut RangeInclusive<u32>,
    to_grab: u32,
    stride: u32,
    full_page_movement: bool,
) -> usize {
    if !page.contains(cursor) {
        let start = *page.start();
        let end = *page.end();
        if *cursor >= end {
            let offset = match full_page_movement {
                false => cursor.saturating_sub(end),
                true => to_grab,
            };
            let end = end.saturating_add(offset);
            *page = (end.saturating_sub(to_grab * stride))..=end;
        } else if *cursor <= start {
            let offset = match full_page_movement {
                false => start.saturating_sub(*cursor),
                true => to_grab,
            };
            let start = start.saturating_sub(offset);
            *page = start..=(start.saturating_add(to_grab * stride));
        }
    };
    (*cursor - page.start()) as usize
}

fn draw_mips_assembly(area: Rect, frame: &mut Frame, tui_state: &mut TuiState, state: &AppState) {
    let area = {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title("+ mips +")
            .title_top(format!(
                "+ {} in {}..{} +",
                hex(tui_state.mips_cursor),
                hex(*tui_state.mips_range.start()),
                hex(*tui_state.mips_range.end())
            ))
            .title_bottom(" <C-j>/<C-k> down/up ".dim())
            .border_style(
                Style::new()
                    .with_theme(&tui_state.theme)
                    .with_focus(tui_state, Focused::Mips),
            );
        let a = block.inner(area);
        frame.render_widget(block, area);
        a
    };
    let [button_area, list_area] =
        Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(area);

    {
        let [button_area] =
            Layout::horizontal([Constraint::Max("jump to pc".len() as u16 + 4)]).areas(button_area);
        let button = Button::new("jump to pc").set_styles(ButtonStyles {
            pressed: tui_state.theme.primary.into(),
            normal:  tui_state.theme.fg.into(),
        });
        frame.render_stateful_widget(button, button_area, &mut tui_state.mips_jump_to_pc_button);
    }

    let offset = compute_infinite_list(
        &mut tui_state.mips_cursor,
        &mut tui_state.mips_range,
        list_area.height as u32,
        4,
        false,
    );
    let mut table_state = TableState::new();
    table_state = table_state.with_selected(offset >> 2);
    let items = tui_state
        .mips_range
        .clone()
        .step_by(0x4)
        .map(|addr| {
            let value = IO::try_read_pure::<u32>(&state.emu, addr)
                .ok()
                .and_then(|value| std::panic::catch_unwind(|| DecodedOp::decode([value])[0]).ok());
            let decoded_op =
                value.unwrap_or(DecodedOp::Illegal(pchan_emu::dynarec_v2::emitters::Illegal));
            (addr, decoded_op)
        })
        .map(|(addr, op)| {
            let style = match op.is_boundary() {
                false => tui_state.theme.fg,
                true => tui_state.theme.primary,
            };
            let mut decoded_op = op.to_string();
            let idx = addr >> 2;
            if addr == state.emu.cpu.pc {
                use std::fmt::Write;
                _ = write!(decoded_op, " <-");
            };
            let style = if idx % 2 == 0 {
                style.darken(0.25)
            } else {
                style
            };
            Row::new([hex(addr).to_string().dim(), decoded_op.into()]).style(style)
        });
    let list = Table::new(items, [Constraint::Length(10), Constraint::Fill(1)])
        .theme(&tui_state.theme)
        .row_highlight_style(
            Style::new()
                .bg(tui_state.theme.primary)
                .bold()
                .italic()
                .fg(tui_state.theme.fg),
        );
    frame.render_stateful_widget(list, list_area, &mut table_state);
}

fn draw_mem(area: Rect, frame: &mut Frame, tui_state: &mut TuiState, state: &AppState) {
    let area = {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title("+ mem +")
            .title_top(" <C-hjkl> navigate ".dim())
            .title_bottom(format!(
                "+ {} in {}..{} +",
                hex(tui_state.mem_cursor),
                hex(*tui_state.mem_range.start()),
                hex(*tui_state.mem_range.end())
            ))
            .border_style(
                Style::new()
                    .with_theme(&tui_state.theme)
                    .with_focus(tui_state, Focused::Mem),
            );
        let a = block.inner(area);
        frame.render_widget(block, area);
        a
    };
    if tui_state.mem_range.start() == tui_state.mem_range.end() {
        tui_state.mem_range =
            *tui_state.mem_range.start()..=(*tui_state.mem_range.start() + area.height as u32 * 16);
    }
    let offset = compute_infinite_list(
        &mut tui_state.mem_cursor,
        &mut tui_state.mem_range,
        area.height as u32 * 16,
        1,
        true,
    );

    let mut table_state = TableState::new();
    let mut row_idx = 0;
    let hexdump = tui_state
        .mem_range
        .clone()
        .step_by(0x4)
        .array_chunks::<4>()
        .map(|addr| {
            row_idx = (row_idx + 1) % 2;
            let start = addr[0];
            let values = addr.map(|addr| {
                let value = IO::try_read_pure::<u32>(&state.emu, addr).unwrap_or(0x0);
                let idx = (addr - start) >> 2;
                let style = if (idx + row_idx) % 2 == 0 {
                    tui_state.theme.fg.darken(0.25)
                } else {
                    tui_state.theme.fg
                };

                value.to_le_bytes().map(|byte| {
                    let style = {
                        let mut style = Style::from(style);
                        if byte == 0 {
                            style = style.dim();
                        }
                        style
                    };
                    hex_pref::<_, false>(byte).to_string().set_style(style)
                })
            });
            Row::new(values.into_iter().flatten())
        });
    table_state = table_state.with_selected_cell((offset >> 4, tui_state.mem_cursor as usize % 16));
    let hexdump_width = 9 * 4;
    let hexdump = Table::new(
        hexdump,
        [
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(3),
        ]
        .repeat(4),
    )
    .theme(&tui_state.theme)
    .cell_highlight_style(Style::new().bg(tui_state.theme.primary).bold().italic())
    .column_spacing(0);
    let [hexdump_area, sep_area, inspect_area] = Layout::horizontal([
        Constraint::Length(hexdump_width),
        Constraint::Max(1),
        Constraint::Fill(1),
    ])
    .areas(area);
    frame.render_stateful_widget(hexdump, hexdump_area, &mut table_state);

    Block::new()
        .borders(Borders::LEFT)
        .border_type(BorderType::LightTripleDashed)
        .render(sep_area, frame.buffer_mut());

    {
        let value_u32 = IO::try_read_pure::<u32>(&state.emu, tui_state.mem_cursor).unwrap_or(0x0);
        let value_u16 = IO::try_read_pure::<u16>(&state.emu, tui_state.mem_cursor).unwrap_or(0x0);
        let value_u8 = IO::try_read_pure::<u8>(&state.emu, tui_state.mem_cursor).unwrap_or(0x0);
        let ascii = IO::try_read_pure::<u32>(&state.emu, tui_state.mem_cursor).unwrap_or(0x0);
        let ascii = ascii
            .to_le_bytes()
            .map(|byte| char::from_u32(byte as u32).unwrap_or_default());
        frame.render_widget(
            ratatui::widgets::List::new([
                ratatui::prelude::Line::from_iter(["u32 ", hex(value_u32).as_str()]),
                ratatui::prelude::Line::from_iter(["u16 ", hex(value_u16).as_str()]),
                ratatui::prelude::Line::from_iter([" u8 ", hex(value_u8).as_str()]),
                ratatui::prelude::Line::from_iter([
                    "ascii ",
                    &format!("{}{}{}{}", ascii[0], ascii[1], ascii[2], ascii[3]),
                ]),
            ]),
            inspect_area,
        );
    }

    if tui_state.jump_to_mem_address_pane.open {
        let [area] = Layout::vertical([Constraint::Length(4)]).areas(area);
        frame.render_widget(Clear, area);
        frame.render_widget(
            EditorView::new(&mut tui_state.jump_to_mem_address_pane.input)
                .theme(
                    EditorTheme::default()
                        .block(Block::bordered().border_type(BorderType::Rounded))
                        .status_line(
                            EditorStatusLine::default().style_mode(
                                Style::new()
                                    .bg(tui_state.theme.primary)
                                    .fg(tui_state.theme.bg),
                            ),
                        ),
                )
                .single_line(true),
            area,
        );
    }
}

fn draw_breakpoints(area: Rect, frame: &mut Frame, tui_state: &mut TuiState, state: &AppState) {
    if tui_state.focused != Focused::Breakpoints {
        tui_state.breakpoints_table.select(None);
    } else if tui_state.breakpoints_table.selected().is_none() {
        tui_state.breakpoints_table.select_first();
    }

    let area = {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title("+ breakpoints +")
            .title_bottom(" <C-jk> navigate • <d> del • <a> add ".dim())
            .border_style(
                Style::new()
                    .with_theme(&tui_state.theme)
                    .with_focus(tui_state, Focused::Breakpoints),
            );
        let a = block.inner(area);
        frame.render_widget(block, area);
        a
    };

    let [breakpoints_area, history_area] =
        Layout::vertical([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(area);

    {
        let breakpoints = state
            .emu
            .dbg
            .breakpoints
            .iter()
            .enumerate()
            .map(|(idx, (addr, brk))| {
                let mut flags = String::with_capacity(4);
                if brk.kind.contains(BreakpointKind::READ) {
                    flags.push('r');
                } else {
                    flags.push('-');
                }
                if brk.kind.contains(BreakpointKind::WRITE) {
                    flags.push('w');
                } else {
                    flags.push('-');
                }
                if brk.kind.contains(BreakpointKind::EXECUTE) {
                    flags.push('x');
                } else {
                    flags.push('-');
                }
                let style = Style::from(tui_state.theme.fg);
                let (onoff, style) = match brk.enabled {
                    true => (" on", style),
                    false => ("off", style.dim()),
                };
                let tooltip = match Some(idx) == tui_state.breakpoints_table.selected() {
                    true => "d: del".set_style(Style::new().fg(tui_state.theme.bg)),
                    false => "".into(),
                };
                Row::new([
                    hex(*addr).to_string().into(),
                    onoff.to_owned().into(),
                    flags.into(),
                    tooltip,
                ])
                .style(style)
            });
        frame.render_stateful_widget(
            Table::new(
                breakpoints,
                [
                    Constraint::Length(10),
                    Constraint::Length(4),
                    Constraint::Length(3),
                    Constraint::Fill(1),
                ],
            )
            .row_highlight_style(Style::new().bg(tui_state.theme.primary).bold().italic())
            .theme(&tui_state.theme),
            breakpoints_area,
            &mut tui_state.breakpoints_table,
        );
    }

    {
        let history = tui_state
            .exec_history
            .iter()
            .rev()
            .take(
                history_area
                    .inner(Margin {
                        horizontal: 1,
                        vertical:   1,
                    })
                    .height as usize,
            )
            .rev()
            .map(|addr| hex(*addr).to_string());
        frame.render_widget(
            List::new(history)
                .block(
                    Block::bordered()
                        .border_type(BorderType::HeavyDoubleDashed)
                        .border_style(Style::new().dim())
                        .title("+ history +"),
                )
                .theme(&tui_state.theme),
            history_area,
        );
    }

    if tui_state.add_breakpoint_pane.open {
        let area = match &tui_state.add_breakpoint_pane.error {
            None => Layout::vertical([Constraint::Max(8)]).areas::<1>(area)[0],
            Some(_) => Layout::vertical([Constraint::Fill(1)]).areas::<1>(area)[0],
        };
        let area = {
            frame.render_widget(Clear, area);
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::new().dim())
                .title_bottom(" <jk/tab> navigate • <ret> confirm • <esc> cancel ".dim())
                .title_top("+ add breakpoint +");

            let inner = block.inner(area);
            frame.render_widget(block, area);
            inner
        };

        let [normal_area, err_area] =
            Layout::vertical([Constraint::Max(8), Constraint::Fill(1)]).areas(area);
        let [addr_input, read_area, write_area, exec_area, done_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(normal_area);
        {
            let mut theme = edtui::EditorTheme::default()
                .hide_status_line()
                .block(Block::new().title_top(format!(
                    "{:?} address: ",
                    tui_state.add_breakpoint_pane.address_input.mode
                )))
                .cursor_style(Style::new().bg(tui_state.theme.primary));
            if tui_state.add_breakpoint_pane.focus != Some(0) {
                theme = theme.hide_cursor();
            }
            let input = EditorView::new(&mut tui_state.add_breakpoint_pane.address_input)
                .single_line(true)
                .theme(theme);
            frame.render_widget(input, addr_input);
        }

        {
            let focused = tui_state.add_breakpoint_pane.focus;
            frame.render_stateful_widget(
                Checkbox::new("read").focus_style_if(&tui_state.theme, focused == Some(1)),
                read_area,
                &mut tui_state.add_breakpoint_pane.checkboxes[0],
            );
            frame.render_stateful_widget(
                Checkbox::new("write").focus_style_if(&tui_state.theme, focused == Some(2)),
                write_area,
                &mut tui_state.add_breakpoint_pane.checkboxes[1],
            );
            frame.render_stateful_widget(
                Checkbox::new("exec").focus_style_if(&tui_state.theme, focused == Some(3)),
                exec_area,
                &mut tui_state.add_breakpoint_pane.checkboxes[2],
            );
            frame.render_widget(
                " Done ".focus_style_if(&tui_state.theme, focused == Some(4)),
                done_area,
            );
        }

        if let Some(err) = &tui_state.add_breakpoint_pane.error {
            use ansi_to_tui::IntoText as _;
            let err = format!("{err:?}");
            let Ok(err) = err.into_text() else {
                frame.render_widget("invalid error.", err_area);
                return;
            };
            frame.render_widget(err, err_area);
        }
    }
}

trait Themed: Styled + Sized {
    fn theme(self, theme: &Theme) -> Self::Item {
        self.set_style(Style::new().with_theme(theme))
    }
    fn focus_style(self, tui_state: &TuiState, value: Focused) -> Self::Item {
        let style = self.style();
        self.set_style(style.with_focus(tui_state, value))
    }
    fn focus_style_if(self, theme: &Theme, cond: bool) -> Self::Item {
        let style = self.style();
        self.set_style(style.with_focus_if(theme, cond))
    }
}

trait StyleExt: Sized {
    fn with_theme(self, theme: &Theme) -> Self;
    fn with_focus(self, tui_state: &TuiState, value: Focused) -> Self {
        self.with_focus_if(&tui_state.theme, tui_state.focused == value)
    }
    fn with_focus_if(self, theme: &Theme, cond: bool) -> Self;
}

impl StyleExt for Style {
    fn with_theme(self, theme: &Theme) -> Self {
        self.fg(theme.fg)
    }
    fn with_focus_if(self, theme: &Theme, cond: bool) -> Self {
        if cond {
            self.fg(theme.primary)
        } else {
            self.with_theme(theme)
        }
    }
}

impl<T: Styled> Themed for T {}

trait ColorExt: Sized {
    const BLACK: Self;
    const WHITE: Self;

    fn lerp(self, other: Self, t: f32) -> Self;
    fn darken(self, t: f32) -> Self {
        self.lerp(Self::BLACK, t)
    }
}

impl ColorExt for u8 {
    const BLACK: Self = 0;
    const WHITE: Self = 255;

    fn lerp(self, other: Self, t: f32) -> Self {
        ((self as f32 * (1.0 - t)) + other as f32 * t) as u8
    }
}

impl ColorExt for Color {
    const BLACK: Self = Self::Rgb(0, 0, 0);

    const WHITE: Self = Self::Rgb(255, 255, 255);

    fn lerp(self, other: Self, t: f32) -> Self {
        if let (Self::Rgb(r0, g0, b0), Self::Rgb(r1, g1, b1)) = (self, other) {
            Self::Rgb(r0.lerp(r1, t), g0.lerp(g1, t), b0.lerp(b1, t))
        } else {
            self
        }
    }
}

impl ColorExt for Option<Color> {
    const BLACK: Self = Some(Color::BLACK);
    const WHITE: Self = Some(Color::WHITE);

    fn lerp(self, other: Self, t: f32) -> Self {
        if let (Some(a), Some(b)) = (self, other) {
            Some(a.lerp(b, t))
        } else {
            self
        }
    }
}

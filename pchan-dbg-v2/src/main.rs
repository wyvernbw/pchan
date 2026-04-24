#![feature(duration_millis_float)]

use std::{
    collections::{HashMap, VecDeque},
    io::stdout,
    ops::{Range, RangeInclusive},
    sync::Arc,
    time::{Duration, Instant},
};

use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture};
use crossterm_simple_event::CrosstermSimpleEvent;
use image::{DynamicImage, RgbaImage};
use miette::{IntoDiagnostic, Result, miette};
use pchan_emu::{
    Emu,
    bootloader::Bootloader,
    cpu::reg_str,
    dynarec_v2::{Dynarec, PipelineV2, emitters::DecodedOp, run_step},
    io::{IO, vblank::VBlank},
};
use pchan_gpu::Renderer;
use pchan_utils::hex;
use rat_imaginary::{ImageState, ImageWidget};
use rat_widget::{
    button::{Button, ButtonState},
    event::{HandleEvent, MouseOnly},
    list::{List, ListState, selection::RowSelection},
    scrolled::Scroll,
};
use ratatui::{
    DefaultTerminal, Frame, Terminal, crossterm,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Styled},
    widgets::{Block, BorderType, Row, Table, TableState, Widget},
};
use wgpu::Extent3d;

use crate::{
    display::{DisplayState, draw_display},
    init::EnvVars,
    lipgloss_colors::LIPGLOSS,
};

pub mod display;
pub mod init;
#[path = "./lipgloss-colors.rs"]
pub mod lipgloss_colors;

fn main() -> Result<()> {
    miette_panic::install(miette_panic::PanicHookArgs::default());
    let env = EnvVars::new()?;
    smol::block_on(run_app(&env))
}

struct AppState {
    emu: Emu,
    gpu: Arc<Renderer>,
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

    mips_jump_to_pc_button: ButtonState,
}

struct Theme {
    fg:      Color,
    primary: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            fg:      Color::from_u32(0xffffff),
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
    let mut tui_state = TuiState {
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

        mips_jump_to_pc_button: ButtonState::new(),
    };
    tui_state.reg_list.select_first();
    state.gpu.clone().start();

    run(|term| {
        let mut pipe = PipelineV2::new(&state.emu);
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
                        handle_event(&state, &mut tui_state, &ev);
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
                        handle_event(&state, &mut tui_state, &ev);
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
                        handle_event(&state, &mut tui_state, &ev);
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
    });

    Ok(())
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

fn handle_event(state: &AppState, tui_state: &mut TuiState, ev: &event::Event) {
    match ev.simple().as_str() {
        "ctrl+c" | "q" => tui_state.quit = true,
        "f" => tui_state.fullscreen = !tui_state.fullscreen,
        " " => {
            tui_state.emu_running = !tui_state.emu_running;
            match tui_state.loop_mode {
                LoopMode::Poll => tui_state.loop_mode = LoopMode::Event,
                LoopMode::Event => tui_state.loop_mode = LoopMode::Poll,
            }
        }
        event => {
            match tui_state.mips_jump_to_pc_button.handle(ev, MouseOnly) {
                rat_widget::event::ButtonOutcome::Continue => {}
                rat_widget::event::ButtonOutcome::Unchanged => {}
                rat_widget::event::ButtonOutcome::Changed => {}
                rat_widget::event::ButtonOutcome::Pressed => {
                    tui_state.mips_cursor = state.emu.cpu.pc;
                }
            };
            match (tui_state.focused, event) {
                (Focused::Preview, "tab" | "j") => tui_state.focused = Focused::Registers,
                (Focused::Preview, "l") => tui_state.focused = Focused::Mips,
                (Focused::Registers, "backtab" | "k") => tui_state.focused = Focused::Preview,
                (Focused::Registers, "l") => tui_state.focused = Focused::Mips,
                (Focused::Registers, "ctrl+k" | "up") => tui_state.reg_list.select_previous(),
                (Focused::Registers, "ctrl+j" | "down") => tui_state.reg_list.select_next(),
                (Focused::Mips, "h") => tui_state.focused = Focused::Registers,
                (Focused::Mips, "ctrl+j") => {
                    tui_state.mips_cursor = tui_state.mips_cursor.saturating_add(4);
                }
                (Focused::Mips, "ctrl+k") => {
                    tui_state.mips_cursor = tui_state.mips_cursor.saturating_sub(4);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

impl TuiState {
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
        .title("+ 🐷🎗️ P-ちゃん dbg (v2) +");
    let area = main_block.inner(frame.area());
    main_block.render(frame.area(), frame.buffer_mut());

    let [h1, h2] =
        Layout::horizontal([Constraint::Ratio(1, 3), Constraint::Ratio(1, 3)]).areas(area);

    let rest = if let Some(img) = &tui_state.current_frame {
        let ar = img.height() * 100 / img.width();
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
                img.width(),
                img.height(),
                frame_time,
                fps
            ))
            .theme(&tui_state.theme)
            .focus_style(tui_state, Focused::Preview);
        let inner_img_area = img_block.inner(img_area);
        img_block.render(img_area, frame.buffer_mut());
        ImageWidget.render(inner_img_area, frame, &mut tui_state.framebuffer);
        rest
    } else {
        h1
    };

    draw_register_viewer(rest, frame, tui_state, state);
    draw_mips_assembly(h2, frame, tui_state, state);
}

fn draw_register_viewer(area: Rect, frame: &mut Frame, tui_state: &mut TuiState, state: &AppState) {
    let area = {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title_top("+ registers +")
            .theme(&tui_state.theme)
            .focus_style(tui_state, Focused::Registers);
        let a = block.inner(area);
        block.render(area, frame.buffer_mut());
        a
    };

    let pc_hex = hex(state.emu.cpu.pc);
    let rows = [Row::new(["pc", pc_hex.as_str()])];
    let regs_hex = state.emu.cpu.gpr.map(hex);
    let rows = rows
        .into_iter()
        .chain((0..32).map(|r| Row::new([reg_str(r), regs_hex[r as usize].as_str()])));
    let table = Table::new(rows, [Constraint::Length(3), Constraint::Length(10)])
        .row_highlight_style(Style::new().bg(tui_state.theme.primary).bold())
        .theme(&tui_state.theme);
    frame.render_stateful_widget(table, area, &mut tui_state.reg_list);
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
            .theme(&tui_state.theme)
            .focus_style(tui_state, Focused::Mips);
        let a = block.inner(area);
        frame.render_widget(block, area);
        a
    };
    let [button_area, list_area] =
        Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(area);

    {
        let button = Button::new("jump to pc")
            .block(Block::bordered().border_type(BorderType::Rounded))
            .armed_style(tui_state.theme.primary);
        frame.render_stateful_widget(button, button_area, &mut tui_state.mips_jump_to_pc_button);
    }

    let to_grab = list_area.height as u32;
    let mut table_state = TableState::new();
    if !tui_state.mips_range.contains(&tui_state.mips_cursor) {
        let start = *tui_state.mips_range.start();
        let end = *tui_state.mips_range.end();
        if tui_state.mips_cursor >= end {
            let offset = tui_state.mips_cursor.saturating_sub(end);
            let end = end.saturating_add(offset);
            tui_state.mips_range = (end.saturating_sub(to_grab * 4))..=end;
        } else if tui_state.mips_cursor <= start {
            let offset = start.saturating_sub(tui_state.mips_cursor);
            let start = start.saturating_sub(offset);
            tui_state.mips_range = start..=(start.saturating_add(to_grab * 4));
        }
    };
    let offset = tui_state
        .mips_cursor
        .saturating_sub(*tui_state.mips_range.start());
    table_state = table_state.with_selected(offset as usize >> 2);
    let items = tui_state
        .mips_range
        .clone()
        .step_by(0x4)
        .map(|addr| {
            let value = IO::try_read_pure::<u32>(&state.emu, addr)
                .ok()
                .and_then(|value| std::panic::catch_unwind(|| DecodedOp::decode([value])[0]).ok());
            (
                addr,
                value.unwrap_or(DecodedOp::Illegal(pchan_emu::dynarec_v2::emitters::Illegal)),
            )
        })
        .map(|(addr, op)| Row::new([hex(addr).to_string(), op.to_string()]));
    let list = Table::new(items, [Constraint::Length(10), Constraint::Fill(1)])
        .theme(&tui_state.theme)
        .row_highlight_style(Style::new().bg(tui_state.theme.primary).bold());
    frame.render_stateful_widget(list, list_area, &mut table_state);
}

trait Themed: Styled + Sized {
    fn theme(self, theme: &Theme) -> Self::Item {
        self.set_style(Style::new().fg(theme.fg))
    }
    fn focus_style(self, tui_state: &TuiState, value: Focused) -> Self::Item {
        if tui_state.focused == value {
            self.set_style(Style::new().fg(tui_state.theme.primary))
        } else {
            self.theme(&tui_state.theme)
        }
    }
}

impl<T: Styled> Themed for T {}

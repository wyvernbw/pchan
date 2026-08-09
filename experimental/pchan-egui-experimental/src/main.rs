#![feature(duration_millis_float)]

use egui_extras::{Size, StripBuilder};
use pchan_bind::ringbuf::StaticRb;
use pchan_bind::ringbuf::traits::{Consumer, RingBuffer};
use pchan_emu::cpu::REG_STR;
use pchan_utils::tracy::{PlotConfiguration, plot_name};
use pchan_utils::{InitTracingArgs, default, tracy};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Color32, FontData, FontDefinitions, Panel, RichText, ScrollArea, Sense, Style, Ui,
};
use eframe::egui_wgpu::{Callback, CallbackTrait};
use miette::{Context, IntoDiagnostic};
use pchan_audio::AudioTask;
use pchan_emu::Emu;
use pchan_emu::run::Runner;

fn main() -> miette::Result<()> {
    pchan_utils::init_tracing(InitTracingArgs {
        panic_hook: false,
        file:       true,
        stdout:     false,
    });
    let native_options = eframe::NativeOptions {
        persist_window: true,
        wgpu_options: eframe::WgpuConfiguration {
            surface: eframe::SurfaceConfig {
                present_mode:                  eframe::wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: Some(2),
            },
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "P-chan gui",
        native_options,
        Box::new(|cc| Ok(Box::new(MyEguiApp::try_new(cc)?))),
    )
    .into_diagnostic()
}

struct MyEguiApp {
    emu:           Emu,
    emu_running:   bool,
    gpu:           Arc<pchan_gpu::Renderer>,
    runner:        Runner,
    frame_instant: Instant,
    frame_idx:     usize,

    game_window_open: bool,
    game_display_cb:  GameDisplayCallback,

    disc_path: Option<PathBuf>,

    pc_history:     StaticRb<u32, 128>,
    register_edits: [String; 34],
}

#[derive(PartialEq)]
enum LoopMode {
    OnEvent,
    GameLoop,
}

fn get_system_fonts(defs: &mut FontDefinitions) {
    fn add_from_character(defs: &mut FontDefinitions, db: &fontdb::Database, test_char: char) {
        let font = db.faces().find_map(|face| {
            db.with_face_data(face.id, |data, index| {
                let font = ttf_parser::Face::parse(data, index).ok()?;
                font.glyph_index(test_char).map(|_| face)
            })
            .flatten()
        });
        if let Some(font) = font {
            db.with_face_data(font.id, |data, _| {
                defs.font_data.insert(
                    font.post_script_name.clone(),
                    FontData::from_owned(data.to_owned()).into(),
                );
                defs.families
                    .get_mut(&egui::FontFamily::Proportional)
                    .unwrap()
                    .push(font.post_script_name.to_owned());
            });
        }
    }
    let mut db = fontdb::Database::new();
    db.load_system_fonts();
    add_from_character(defs, &db, 'あ');
    add_from_character(defs, &db, 'a');
}

impl MyEguiApp {
    fn try_new(cc: &eframe::CreationContext<'_>) -> miette::Result<Self> {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_global_style.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        let mut fonts = FontDefinitions::empty();
        fonts.font_data.insert(
            "GeistMono".into(),
            FontData::from_static(include_bytes!("../assets/GeistMono[wght].ttf")).into(),
        );
        fonts.font_data.insert(
            "Geist".into(),
            FontData::from_static(include_bytes!("../assets/Geist[wght].ttf")).into(),
        );
        // fonts
        //     .families
        //     .get_mut(&egui::FontFamily::Proportional)
        //     .unwrap()
        //     .push("GeistMono".into());
        // fonts
        //     .families
        //     .get_mut(&egui::FontFamily::Proportional)
        //     .unwrap()
        //     .push("Geist".into());
        get_system_fonts(&mut fonts);
        cc.egui_ctx.set_fonts(fonts);

        let mut emu = Emu::new();
        let bios_path = std::env::var("PCHAN_BIOS")
            .into_diagnostic()
            .wrap_err("PCHAN_BIOS env var not set")?
            .parse::<PathBuf>()
            .wrap_err("invalid bios path")?;
        emu.set_bios_path(bios_path);
        emu.load_bios().into_diagnostic()?;
        emu.cpu.jump_to_bios();

        let wgpu = cc.wgpu_render_state.as_ref().expect("expected wgpu");
        let mut gpu = pchan_executor::block_on(pchan_gpu::Renderer::from_wgpu(
            wgpu.adapter.clone(),
            wgpu.device.clone(),
            wgpu.queue.clone(),
            eframe::wgpu::TextureFormat::Bgra8Unorm,
        ))
        .into_diagnostic()?;
        gpu.connect_emu(&mut emu);
        gpu.display_uniforms.lock().unwrap().dp_srgb = false;
        let gpu = Arc::new(gpu);
        gpu.clone().start();

        let mut audio_task = AudioTask::new()?;
        pchan_bind::bind_audio(&mut audio_task, &mut emu);
        let audio_stream = audio_task.start()?;
        std::mem::forget(audio_stream);

        let game_display_cb = GameDisplayCallback { rd: gpu.clone() };

        emu.tracy.plot_config(
            plot_name!("jit_cache_rate"),
            PlotConfiguration::default().format(tracy::PlotFormat::Percentage),
        );
        Ok(Self {
            emu,
            emu_running: false,
            gpu,
            runner: Runner::new().with_config(pchan_emu::run::RunnerConfig {
                // force_mode: Some(pchan_emu::run::RunnerMode::Dynarec),
                force_mode: Some(pchan_emu::run::RunnerMode::Interpreter),
            }),
            frame_instant: Instant::now(),
            frame_idx: 0,
            game_window_open: true,
            game_display_cb,
            disc_path: None,
            pc_history: StaticRb::default(),

            register_edits: core::array::from_fn(|_| default()),
        })
    }

    fn loop_mode(&self) -> LoopMode {
        match self.emu_running {
            true => LoopMode::GameLoop,
            false => LoopMode::OnEvent,
        }
    }

    fn open_disc(&mut self, path: &Path, streamed: bool) -> miette::Result<()> {
        let fsm = self.emu.open_disc(path, streamed).into_diagnostic()?;
        self.emu
            .advance_open_disc(path, fsm, streamed)
            .into_diagnostic()?;
        Ok(())
    }
}

impl eframe::App for MyEguiApp {
    fn persist_egui_memory(&self) -> bool {
        false
    }

    fn on_exit(&mut self) {}

    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.frame_instant = Instant::now();
        if self.emu_running {
            while !self.emu.consume_vblank_signal() {
                self.pc_history.push_overwrite(self.emu.cpu.pc);
                self.runner.execute(&mut self.emu);
            }
            let blocks_compiled = self.emu.stats.pop_frame_blocks_compiled();
            tracy::plot!("jit_blocks_compiled", blocks_compiled as f64);
            let blocks_ran = self.emu.stats.pop_frame_blocks_ran();
            tracy::plot!("jit_blocks_ran", blocks_ran as f64);
            let cache_rate = (1.0 - blocks_compiled as f64 / blocks_ran as f64) * 100.;
            tracy::plot!("jit_cache_rate", cache_rate);
        }
        self.frame_idx += 1;
    }
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let frame_time = self.frame_instant.elapsed();
        let frame_time_ms = frame_time.as_millis_f32();
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("Hello World!");
            ui.label("This is my incredible emulator, do you like it?");
            ui.radio_value(&mut self.emu_running, false, "paused");
            ui.radio_value(&mut self.emu_running, true, "running");

            let btn_text = self
                .disc_path
                .as_ref()
                .map(|p| p.to_string_lossy())
                .unwrap_or("Open disc...".into());
            if ui.button(btn_text).clicked()
                && let Some(path) = rfd::FileDialog::new().pick_file()
            {
                match self.open_disc(&path, true) {
                    Ok(_) => {
                        self.disc_path = Some(path);
                    }
                    Err(err) => {
                        ui.label(format!("Error: {err}"));
                    }
                };
            }

            if self.emu_running {
                ui.label(format!("frame #{}", self.frame_idx));
                ui.label(format!("frame time: {}ms", frame_time_ms));
                ui.label(format!("fake fps: {}", 1000. / frame_time_ms));
            }
            egui::Window::new("Game")
                .default_width(320.0)
                .default_height(240.0)
                .constrain(true)
                .open(&mut self.game_window_open)
                .resizable(true)
                .show(ui, |ui| {
                    let (res, painter) = ui.allocate_painter(ui.available_size(), Sense::empty());
                    painter.add(Callback::new_paint_callback(
                        res.rect,
                        self.game_display_cb.clone(),
                    ))
                });
            // TODO: handle errors
            self.cpu_info(ui).unwrap();
        });

        if let Some(window) = frame.winit_window() {
            match self.loop_mode() {
                LoopMode::OnEvent => {}
                LoopMode::GameLoop => {
                    let sleep_for =
                        Duration::from_micros(16666).saturating_sub(self.frame_instant.elapsed());
                    spin_sleep::sleep(sleep_for);
                    window.request_redraw()
                }
            }
        }
    }
}

#[derive(Clone)]
struct GameDisplayCallback {
    rd: Arc<pchan_gpu::Renderer>,
}

impl CallbackTrait for GameDisplayCallback {
    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        let rect = info.clip_rect_in_pixels();
        {
            self.rd.display_uniforms.lock().unwrap().screen_rect =
                pchan_gpu::glam::U16Vec2::new(rect.width_px as u16, rect.height_px as u16);
        }
        self.rd.draw_display(render_pass);
    }
}

impl MyEguiApp {
    fn cpu_info(&mut self, ui: &mut Ui) -> miette::Result<()> {
        egui::Window::new("cpu info")
            .default_size([400., 400.])
            .resizable(false)
            .show(ui, |ui| {
                StripBuilder::new(ui)
                    .sizes(Size::relative(0.5), 2)
                    .horizontal(|mut strip| {
                        strip.cell(|ui| {
                            ScrollArea::vertical()
                                .auto_shrink([true, true])
                                .show(ui, |ui| {
                                    egui::Grid::new("cpu_grid")
                                        .num_columns(2)
                                        .striped(true)
                                        .show(ui, |ui| {
                                            ui.vertical(|ui| {
                                                ui.label("pc");
                                                ui.label(format!(
                                                    "{}",
                                                    pchan_utils::hex(self.emu.cpu.pc)
                                                ));
                                                ui.end_row();
                                                let mut name = String::new();
                                                for (reg, reg_str) in
                                                    REG_STR.iter().enumerate().take(2)
                                                {
                                                    use core::fmt::Write;

                                                    write!(name, "${reg_str}").expect("fmt error");
                                                    ui.label(&name);
                                                    self.register_edits[reg].clear();
                                                    self.register_edits[reg].push_str(reg_str);
                                                    let res = ui.text_edit_singleline(
                                                        &mut self.register_edits[reg],
                                                    );
                                                    ui.end_row();

                                                    name.clear();
                                                }
                                            });
                                        });
                                });
                        });
                        strip.cell(|ui| {
                            ui.vertical(|ui| {
                                ui.heading("PC Log");
                                ScrollArea::vertical()
                                    .auto_shrink([true, true])
                                    .show(ui, |ui| {
                                        for &pc in self.pc_history.iter().rev() {
                                            ui.label(format!("{}", pchan_utils::hex(pc)));
                                        }
                                    });
                            });
                        });
                    });
                // ui.horizontal(|ui| {
                // })
            });
        Ok(())
    }
}

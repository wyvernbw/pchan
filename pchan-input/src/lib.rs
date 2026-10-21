use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use pchan_bind::input::{InputEvent, PchanButton};
use pchan_emu::Emu;
use pchan_emu::io::sio::Sio0Port;
use sdl2::controller::{Axis, Button as SdlButton, GameController};
use sdl2::event::Event;
use sdl2::{EventPump, GameControllerSubsystem};

pub struct Input {
    events:      EventPump,
    controllers: HashMap<u32, GameController>,
    ports:       HashMap<Sio0Port, u32>,
    subsystem:   GameControllerSubsystem,
}

impl Input {
    #[must_use]
    pub fn new() -> Self {
        sdl2::hint::set("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");

        let sdl = sdl2::init().unwrap();
        let game_controller_subsystem = sdl.game_controller().unwrap();

        let sdl_events = sdl.event_pump().unwrap();
        Self {
            events:      sdl_events,
            ports:       HashMap::new(),
            controllers: HashMap::new(),
            subsystem:   game_controller_subsystem,
        }
    }

    pub fn drive_gamepads(&mut self, emu: &mut Emu) {
        while let Some(event) = self.events.poll_event() {
            match event {
                Event::Quit { .. } => break,
                Event::ControllerDeviceAdded {
                    timestamp: _,
                    which: index,
                } => {
                    match self.subsystem.open(index) {
                        Err(err) => println!("failed to open controller {index}: {err}"),
                        Ok(controller) => {
                            let id = controller.instance_id();
                            println!("controller added: {id}, {}", controller.name());
                            self.controllers.insert(id, controller);
                            if let Entry::Vacant(e) = self.ports.entry(Sio0Port::Port1) {
                                e.insert(id);
                                emu.sio.sio0ports.port_mut(Sio0Port::Port1).joypad.plug_in();
                            }
                        }
                    };
                }
                Event::ControllerDeviceRemoved {
                    timestamp: _,
                    which: id,
                } => {
                    self.controllers.remove(&id);
                    self.ports.retain(|_, v| *v != id);
                }
                _ => {}
            }
            if let Some((event, which)) = sdl2_to_pchan_input(event) {
                for (port, id) in &self.ports {
                    if *id == which {
                        tracing::info!(?event);
                        emu.send_input_event(event, *port);
                    }
                }
            }
        }
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

fn sdl2_button_to_pchan_button(button: SdlButton) -> Option<PchanButton> {
    match button {
        SdlButton::A => Some(PchanButton::X),
        SdlButton::B => Some(PchanButton::Circle),
        SdlButton::X => Some(PchanButton::Square),
        SdlButton::Y => Some(PchanButton::Triangle),
        SdlButton::Guide => Some(PchanButton::Select),
        SdlButton::Start => Some(PchanButton::Start),
        SdlButton::LeftStick => Some(PchanButton::L3),
        SdlButton::RightStick => Some(PchanButton::R3),
        SdlButton::LeftShoulder => Some(PchanButton::L1),
        SdlButton::RightShoulder => Some(PchanButton::R1),
        SdlButton::DPadUp => Some(PchanButton::DpadUp),
        SdlButton::DPadDown => Some(PchanButton::DpadDown),
        SdlButton::DPadLeft => Some(PchanButton::DpadLeft),
        SdlButton::DPadRight => Some(PchanButton::DpadRight),
        _ => None,
    }
}

fn sdl2_axis_to_pchan_button(axis: Axis, value: i16) -> Option<InputEvent> {
    match axis {
        Axis::TriggerLeft => Some(PchanButton::L2),
        Axis::TriggerRight => Some(PchanButton::R2),
        _ => None,
    }
    .map(|btn| match value {
        ..=0 => InputEvent::Release(btn),
        1.. => InputEvent::Press(btn),
    })
}

#[must_use]
pub fn sdl2_to_pchan_input(sdl2event: Event) -> Option<(InputEvent, u32)> {
    match sdl2event {
        Event::ControllerAxisMotion {
            timestamp: _,
            which,
            axis,
            value,
        } => sdl2_axis_to_pchan_button(axis, value).map(|result| (result, which)),
        Event::ControllerButtonDown {
            timestamp: _,
            which,
            button,
        } => sdl2_button_to_pchan_button(button).map(|btn| (InputEvent::Press(btn), which)),
        Event::ControllerButtonUp {
            timestamp: _,
            which,
            button,
        } => sdl2_button_to_pchan_button(button).map(|btn| (InputEvent::Release(btn), which)),
        _ => None,
    }
}

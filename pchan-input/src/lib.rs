use std::collections::{HashMap, HashSet, hash_map::Entry};

use pchan_bind::input::{InputEvent, PchanButton};
use pchan_emu::{
    Bus, Emu,
    io::sio::{Sio0Port, joypad::InputEvents},
};
use sdl2::{
    EventPump,
    controller::{Axis, Button as SdlButton, GameController},
    event::Event,
};

pub struct Input {
    events:      EventPump,
    gamepads:    HashSet<u32>,
    controllers: Vec<GameController>,
    ports:       HashMap<Sio0Port, u32>,
}

impl Input {
    pub fn new() -> Self {
        let sdl = sdl2::init().unwrap();
        let game_controller_subsystem = sdl.game_controller().unwrap();

        let available = game_controller_subsystem
            .num_joysticks()
            .map_err(|e| format!("can't enumerate joysticks: {}", e))
            .unwrap();
        let mut controllers = (0..available)
            .flat_map(|id| {
                if !game_controller_subsystem.is_game_controller(id) {
                    return None;
                }

                println!("Attempting to open controller {}", id);

                match game_controller_subsystem.open(id) {
                    Ok(c) => {
                        // We managed to find and open a game controller,
                        // exit the loop
                        println!("Success: opened \"{}\"", c.name());
                        Some(c)
                    }
                    Err(e) => {
                        println!("failed: {:?}", e);
                        None
                    }
                }
            })
            .collect();

        let sdl_events = sdl.event_pump().unwrap();
        Self {
            events: sdl_events,
            gamepads: HashSet::new(),
            ports: HashMap::new(),
            controllers,
        }
    }

    pub fn drive_gamepads(&mut self, emu: &mut Emu) {
        while let Some(event) = self.events.poll_event() {
            match event {
                Event::ControllerDeviceAdded {
                    timestamp: _,
                    which,
                } => {
                    self.gamepads.insert(which);
                    if let Entry::Vacant(e) = self.ports.entry(Sio0Port::Port1) {
                        e.insert(which);
                        emu.sio_mut()
                            .sio0ports
                            .port_mut(Sio0Port::Port1)
                            .joypad
                            .plug_in();
                    }
                }
                Event::ControllerDeviceRemoved {
                    timestamp: _,
                    which,
                } => {
                    self.gamepads.remove(&which);
                }
                _ => {}
            }
            if let Some((event, which)) = sdl2_to_pchan_input(event) {
                for (port, gamepad) in self.ports.iter() {
                    if *gamepad == which {
                        emu.send_input_event(event, *port);
                    }
                }
            }
        }
    }

    pub fn gamepads(&self) -> &HashSet<u32> {
        &self.gamepads
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

fn sdl2_button_to_pchan_button(button: &SdlButton) -> Option<PchanButton> {
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

fn sdl2_axis_to_pchan_button(axis: &Axis, value: i16) -> Option<InputEvent> {
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

pub fn sdl2_to_pchan_input(sdl2event: Event) -> Option<(InputEvent, u32)> {
    match sdl2event {
        Event::ControllerAxisMotion {
            timestamp: _,
            which,
            axis,
            value,
        } => sdl2_axis_to_pchan_button(&axis, value).map(|result| (result, which)),
        Event::ControllerButtonDown {
            timestamp: _,
            which,
            button,
        } => sdl2_button_to_pchan_button(&button).map(|btn| (InputEvent::Press(btn), which)),
        Event::ControllerButtonUp {
            timestamp: _,
            which,
            button,
        } => sdl2_button_to_pchan_button(&button).map(|btn| (InputEvent::Release(btn), which)),
        _ => None,
    }
}

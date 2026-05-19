use std::collections::{HashMap, HashSet, hash_map::Entry};

use gilrs_core::Gilrs;
use pchan_emu::{
    Bus, Emu,
    io::sio::{Sio0Port, joypad::InputEvents},
};

pub struct Input {
    gilrs:    Gilrs,
    gamepads: HashSet<usize>,
    ports:    HashMap<Sio0Port, usize>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            gilrs:    Gilrs::new().expect("failed to init gamepads"),
            gamepads: HashSet::new(),
            ports:    HashMap::new(),
        }
    }

    pub fn drive_gamepads(&mut self, emu: &mut Emu) {
        while let Some(ev) = self.gilrs.next_event() {
            match ev.event {
                gilrs_core::EventType::Connected => {
                    self.gamepads.insert(ev.id);
                    if let Entry::Vacant(e) = self.ports.entry(Sio0Port::Port1) {
                        e.insert(ev.id);
                        emu.sio_mut()
                            .sio0ports
                            .port_mut(Sio0Port::Port1)
                            .joypad
                            .plug_in();
                    }
                }
                gilrs_core::EventType::Disconnected => {
                    self.gamepads.remove(&ev.id);
                }
                _ => {}
            }
            for (port, gamepad) in self.ports.iter() {
                if *gamepad == ev.id {
                    emu.send_input_event(ev, *port);
                }
            }
        }
    }

    pub fn gamepads(&self) -> &HashSet<usize> {
        &self.gamepads
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

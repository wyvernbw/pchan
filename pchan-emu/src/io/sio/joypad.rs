use bitbybit::{bitenum, bitfield};
use pchan_bind::input::{InputEvent, PchanButton};

use crate::Emu;
use crate::io::sio::{Peripheral, Sio0Port, TxWriteResult};

use super::Sio0Rx;

#[derive(Default, derive_more::Debug, Clone)]
pub struct Joypad {
    pub connected: bool,
    state:         JoypadState,
    id:            ControllerId,
    switches:      DigitalSwitches,
}

#[derive(Default, derive_more::Debug, Clone)]
enum JoypadState {
    #[default]
    Idle,
    WaitTap,
    WaitMot1,
    WaitMot2,
}

/// 0x5a hi bits is implied
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
#[repr(u8)]
enum ControllerId {
    #[default]
    DigitalPad = 0x41,
}

impl Joypad {
    pub fn plug_in(&mut self) -> &mut Self {
        self.connected = true;
        self
    }
}

impl Peripheral for Joypad {
    fn on_tx_write(&mut self, byte: u8, rx: &mut Sio0Rx) -> TxWriteResult {
        match (&self.state, byte) {
            (JoypadState::Idle, 0x42) => {
                rx.push(self.id as u8);
                self.state = JoypadState::WaitTap;
                TxWriteResult::Ok
            }
            (JoypadState::Idle, _) => TxWriteResult::TransferFinished,
            (JoypadState::WaitTap, _) => {
                rx.push(0x5a);
                self.state = JoypadState::WaitMot1;
                TxWriteResult::Ok
            }
            (JoypadState::WaitMot1, _) => {
                rx.push(self.switches.lower());
                self.state = JoypadState::WaitMot2;
                TxWriteResult::Ok
            }
            (JoypadState::WaitMot2, _) => {
                rx.push(self.switches.upper());
                match self.id {
                    ControllerId::DigitalPad => {
                        self.state = JoypadState::Idle;
                        TxWriteResult::TransferFinished
                    }
                }
            }
        }
    }
}

#[bitenum(u1, exhaustive = true)]
#[derive(Default, Debug)]
pub enum ButtonState {
    Pressed  = 0x0,
    #[default]
    Released = 0x1,
}

/// ```plaintext
///  0   Select Button    (0=Pressed, 1=Released)
///  1   L3/Joy-button    (0=Pressed, 1=Released/None/Disabled) ;analog mode only
///  2   R3/Joy-button    (0=Pressed, 1=Released/None/Disabled) ;analog mode only
///  3   Start Button     (0=Pressed, 1=Released)
///  4   Joypad Up        (0=Pressed, 1=Released)
///  5   Joypad Right     (0=Pressed, 1=Released)
///  6   Joypad Down      (0=Pressed, 1=Released)
///  7   Joypad Left      (0=Pressed, 1=Released)
///  8   L2 Button        (0=Pressed, 1=Released) (Lower-left shoulder)
///  9   R2 Button        (0=Pressed, 1=Released) (Lower-right shoulder)
///  10  L1 Button        (0=Pressed, 1=Released) (Upper-left shoulder)
///  11  R1 Button        (0=Pressed, 1=Released) (Upper-right shoulder)
///  12  /\ Button        (0=Pressed, 1=Released) (Triangle, upper button)
///  13  () Button        (0=Pressed, 1=Released) (Circle, right button)
///  14  >< Button        (0=Pressed, 1=Released) (Cross, lower button)
///  15  [] Button        (0=Pressed, 1=Released) (Square, left button)
/// ```
#[bitfield(u16, debug, default = 0xffff)]
pub struct DigitalSwitches {
    #[bits(0..=7, r)]
    lower: u8,
    #[bits(8..=15, r)]
    upper: u8,

    #[bit(0, rw)]
    select:     ButtonState,
    #[bit(1, rw)]
    l3:         ButtonState,
    #[bit(2, rw)]
    r3:         ButtonState,
    #[bit(3, rw)]
    start:      ButtonState,
    #[bit(4, rw)]
    dpad_up:    ButtonState,
    #[bit(5, rw)]
    dpad_right: ButtonState,
    #[bit(6, rw)]
    dpad_down:  ButtonState,
    #[bit(7, rw)]
    dpad_left:  ButtonState,
    #[bit(8, rw)]
    l2:         ButtonState,
    #[bit(9, rw)]
    r2:         ButtonState,
    #[bit(10, rw)]
    l1:         ButtonState,
    #[bit(11, rw)]
    r1:         ButtonState,
    #[bit(12, rw)]
    triangle:   ButtonState,
    #[bit(13, rw)]
    circle:     ButtonState,
    #[bit(14, rw)]
    x:          ButtonState,
    #[bit(15, rw)]
    square:     ButtonState,
}

impl DigitalSwitches {
    pub fn press(&mut self, code: PchanButton) {
        self.set_state(code, ButtonState::Pressed);
    }
    pub fn release(&mut self, code: PchanButton) {
        self.set_state(code, ButtonState::Released);
    }
    pub fn set_state(&mut self, code: PchanButton, state: ButtonState) {
        match code {
            // TODO: all buttons
            PchanButton::DpadDown => self.set_dpad_down(state),
            PchanButton::DpadUp => self.set_dpad_up(state),
            PchanButton::DpadLeft => self.set_dpad_left(state),
            PchanButton::DpadRight => self.set_dpad_right(state),
            PchanButton::Triangle => self.set_triangle(state),
            PchanButton::X => self.set_x(state),
            PchanButton::Circle => self.set_circle(state),
            PchanButton::Square => self.set_square(state),
            _ => {}
        }
    }
}

impl Emu {
    pub fn send_input_event(&mut self, event: InputEvent, port: Sio0Port) {
        match event {
            InputEvent::Press(btn) => {
                self.sio_mut()
                    .sio0ports
                    .port_mut(port)
                    .joypad
                    .switches
                    .press(btn);
            }
            InputEvent::Release(btn) => {
                self.sio_mut()
                    .sio0ports
                    .port_mut(port)
                    .joypad
                    .switches
                    .release(btn);
            }
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PchanButton {
    Select,
    L3,
    R3,
    Start,
    DpadUp,
    DpadRight,
    DpadDown,
    DpadLeft,
    L2,
    R2,
    L1,
    R1,
    Triangle,
    Circle,
    X,
    Square,
}

#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Press(PchanButton),
    Release(PchanButton),
}

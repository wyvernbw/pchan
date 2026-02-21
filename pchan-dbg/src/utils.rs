#[macro_export]
macro_rules! key {
    // Variant with no arguments, e.g. Enter, Esc
    ($code:ident) => {
        ratatui::crossterm::event::Event::Key(ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code,
            kind: ratatui::crossterm::event::KeyEventKind::Press,
            modifiers: ratatui::crossterm::event::KeyModifiers::NONE,
            ..
        })
    };
    // Variant with arguments, e.g. Char('x'), Char(_)
    ($code:ident ( $($arg:tt)* )) => {
        ratatui::crossterm::event::Event::Key(
            ratatui::crossterm::event::KeyEvent {
                code: ratatui::crossterm::event::KeyCode::$code($($arg)*),
                kind: ratatui::crossterm::event::KeyEventKind::Press,
                modifiers: ratatui::crossterm::event::KeyModifiers::NONE,
                ..
            }
        )
    };

    // Variant with no arguments and optional modifiers
    ($code:ident, $mods:pat ) => {
        ratatui::crossterm::event::Event::Key(ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code,
            kind: ratatui::crossterm::event::KeyEventKind::Press,
            modifiers: $mods,
            ..
        })
    };
    ($code:ident ( $($arg:tt)* ), $mods:pat) => {
        ratatui::crossterm::event::Event::Key(ratatui::crossterm::event::KeyEvent {
            code: ratatui::crossterm::event::KeyCode::$code($($arg)*),
            kind: ratatui::crossterm::event::KeyEventKind::Press,
            modifiers: $mods,
            ..
        })
    };
}

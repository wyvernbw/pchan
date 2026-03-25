use manatui::prelude::{Color, Style};

pub const LIPGLOSS: [[Color; 14]; 8] = [
    [
        Color::from_u32(0xf25d94),
        Color::from_u32(0xf36c94),
        Color::from_u32(0xf37994),
        Color::from_u32(0xf48693),
        Color::from_u32(0xf49293),
        Color::from_u32(0xf59e92),
        Color::from_u32(0xf5a991),
        Color::from_u32(0xf5b490),
        Color::from_u32(0xf4bf8f),
        Color::from_u32(0xf4ca8e),
        Color::from_u32(0xf3d58c),
        Color::from_u32(0xf2df8a),
        Color::from_u32(0xf1ea88),
        Color::from_u32(0xeff585),
    ],
    [
        Color::from_u32(0xe559a1),
        Color::from_u32(0xe668a1),
        Color::from_u32(0xe676a1),
        Color::from_u32(0xe783a0),
        Color::from_u32(0xe790a0),
        Color::from_u32(0xe79b9f),
        Color::from_u32(0xe7a79f),
        Color::from_u32(0xe7b29e),
        Color::from_u32(0xe6bd9d),
        Color::from_u32(0xe6c89b),
        Color::from_u32(0xe5d39a),
        Color::from_u32(0xe4de98),
        Color::from_u32(0xe3e996),
        Color::from_u32(0xe1f393),
    ],
    [
        Color::from_u32(0xd855ad),
        Color::from_u32(0xd865ad),
        Color::from_u32(0xd873ad),
        Color::from_u32(0xd881ac),
        Color::from_u32(0xd98dac),
        Color::from_u32(0xd999ab),
        Color::from_u32(0xd8a5ab),
        Color::from_u32(0xd8b0aa),
        Color::from_u32(0xd8bca9),
        Color::from_u32(0xd7c7a7),
        Color::from_u32(0xd6d2a6),
        Color::from_u32(0xd5dda4),
        Color::from_u32(0xd4e8a2),
        Color::from_u32(0xd2f3a0),
    ],
    [
        Color::from_u32(0xca50b9),
        Color::from_u32(0xca61b8),
        Color::from_u32(0xca70b8),
        Color::from_u32(0xca7eb7),
        Color::from_u32(0xc98bb7),
        Color::from_u32(0xc997b6),
        Color::from_u32(0xc9a3b5),
        Color::from_u32(0xc8afb5),
        Color::from_u32(0xc8bab4),
        Color::from_u32(0xc7c5b2),
        Color::from_u32(0xc6d0b1),
        Color::from_u32(0xc5dcaf),
        Color::from_u32(0xc3e7ad),
        Color::from_u32(0xc1f2ab),
    ],
    [
        Color::from_u32(0xbc4cc4),
        Color::from_u32(0xbb5ec3),
        Color::from_u32(0xba6ec2),
        Color::from_u32(0xba7cc2),
        Color::from_u32(0xb989c1),
        Color::from_u32(0xb895c0),
        Color::from_u32(0xb8a1bf),
        Color::from_u32(0xb7adbf),
        Color::from_u32(0xb6b8be),
        Color::from_u32(0xb5c4bc),
        Color::from_u32(0xb4cfbb),
        Color::from_u32(0xb3dab9),
        Color::from_u32(0xb1e6b7),
        Color::from_u32(0xaff1b5),
    ],
    [
        Color::from_u32(0xac48d0),
        Color::from_u32(0xaa5bce),
        Color::from_u32(0xa96bcd),
        Color::from_u32(0xa879cc),
        Color::from_u32(0xa787cb),
        Color::from_u32(0xa693ca),
        Color::from_u32(0xa59fc9),
        Color::from_u32(0xa4abc8),
        Color::from_u32(0xa3b7c7),
        Color::from_u32(0xa2c2c6),
        Color::from_u32(0xa0cec4),
        Color::from_u32(0x9fd9c3),
        Color::from_u32(0x9de5c1),
        Color::from_u32(0x9af0bf),
    ],
    [
        Color::from_u32(0x9a43dd),
        Color::from_u32(0x9758da),
        Color::from_u32(0x9569d7),
        Color::from_u32(0x9377d6),
        Color::from_u32(0x9285d4),
        Color::from_u32(0x9092d3),
        Color::from_u32(0x8f9ed2),
        Color::from_u32(0x8eaad1),
        Color::from_u32(0x8cb6d0),
        Color::from_u32(0x8bc1cf),
        Color::from_u32(0x89cdcd),
        Color::from_u32(0x87d8cc),
        Color::from_u32(0x84e4ca),
        Color::from_u32(0x82efc7),
    ],
    [
        Color::from_u32(0x843fec),
        Color::from_u32(0x8055e7),
        Color::from_u32(0x7c67e3),
        Color::from_u32(0x7a76e0),
        Color::from_u32(0x7883de),
        Color::from_u32(0x7690dc),
        Color::from_u32(0x749cdb),
        Color::from_u32(0x72a9da),
        Color::from_u32(0x70b4d8),
        Color::from_u32(0x6dc0d7),
        Color::from_u32(0x6bccd6),
        Color::from_u32(0x68d7d4),
        Color::from_u32(0x65e3d2),
        Color::from_u32(0x61eed0),
    ],
];

macro_rules! lipgloss_methods {
    ($($name:ident, $bg_name:ident, $row:expr, $col:expr);* $(;)?) => {
        #[allow(dead_code)]
        pub trait LipglossStyle {
            $(
                fn $name(self) -> Self;
                fn $bg_name(self) -> Self;
            )*
        }

        impl LipglossStyle for Style {
            $(
                fn $name(self) -> Self {
                    self.fg(LIPGLOSS[$row][$col])
                }
                fn $bg_name(self) -> Self {
                    self.bg(LIPGLOSS[$row][$col]).fg(Color::from_u32(0xffffff))
                }
            )*
        }
    };
}

lipgloss_methods! {
    c0000, on_c0000, 0, 0;
    c0001, on_c0001, 0, 1;
    c0002, on_c0002, 0, 2;
    c0003, on_c0003, 0, 3;
    c0004, on_c0004, 0, 4;
    c0005, on_c0005, 0, 5;
    c0006, on_c0006, 0, 6;
    c0007, on_c0007, 0, 7;
    c0008, on_c0008, 0, 8;
    c0009, on_c0009, 0, 9;
    c0010, on_c0010, 0, 10;
    c0011, on_c0011, 0, 11;
    c0012, on_c0012, 0, 12;
    c0013, on_c0013, 0, 13;
    c0100, on_c0100, 1, 0;
    c0101, on_c0101, 1, 1;
    c0102, on_c0102, 1, 2;
    c0103, on_c0103, 1, 3;
    c0104, on_c0104, 1, 4;
    c0105, on_c0105, 1, 5;
    c0106, on_c0106, 1, 6;
    c0107, on_c0107, 1, 7;
    c0108, on_c0108, 1, 8;
    c0109, on_c0109, 1, 9;
    c0110, on_c0110, 1, 10;
    c0111, on_c0111, 1, 11;
    c0112, on_c0112, 1, 12;
    c0113, on_c0113, 1, 13;
    c0200, on_c0200, 2, 0;
    c0201, on_c0201, 2, 1;
    c0202, on_c0202, 2, 2;
    c0203, on_c0203, 2, 3;
    c0204, on_c0204, 2, 4;
    c0205, on_c0205, 2, 5;
    c0206, on_c0206, 2, 6;
    c0207, on_c0207, 2, 7;
    c0208, on_c0208, 2, 8;
    c0209, on_c0209, 2, 9;
    c0210, on_c0210, 2, 10;
    c0211, on_c0211, 2, 11;
    c0212, on_c0212, 2, 12;
    c0213, on_c0213, 2, 13;
    c0300, on_c0300, 3, 0;
    c0301, on_c0301, 3, 1;
    c0302, on_c0302, 3, 2;
    c0303, on_c0303, 3, 3;
    c0304, on_c0304, 3, 4;
    c0305, on_c0305, 3, 5;
    c0306, on_c0306, 3, 6;
    c0307, on_c0307, 3, 7;
    c0308, on_c0308, 3, 8;
    c0309, on_c0309, 3, 9;
    c0310, on_c0310, 3, 10;
    c0311, on_c0311, 3, 11;
    c0312, on_c0312, 3, 12;
    c0313, on_c0313, 3, 13;
    c0400, on_c0400, 4, 0;
    c0401, on_c0401, 4, 1;
    c0402, on_c0402, 4, 2;
    c0403, on_c0403, 4, 3;
    c0404, on_c0404, 4, 4;
    c0405, on_c0405, 4, 5;
    c0406, on_c0406, 4, 6;
    c0407, on_c0407, 4, 7;
    c0408, on_c0408, 4, 8;
    c0409, on_c0409, 4, 9;
    c0410, on_c0410, 4, 10;
    c0411, on_c0411, 4, 11;
    c0412, on_c0412, 4, 12;
    c0413, on_c0413, 4, 13;
    c0500, on_c0500, 5, 0;
    c0501, on_c0501, 5, 1;
    c0502, on_c0502, 5, 2;
    c0503, on_c0503, 5, 3;
    c0504, on_c0504, 5, 4;
    c0505, on_c0505, 5, 5;
    c0506, on_c0506, 5, 6;
    c0507, on_c0507, 5, 7;
    c0508, on_c0508, 5, 8;
    c0509, on_c0509, 5, 9;
    c0510, on_c0510, 5, 10;
    c0511, on_c0511, 5, 11;
    c0512, on_c0512, 5, 12;
    c0513, on_c0513, 5, 13;
    c0600, on_c0600, 6, 0;
    c0601, on_c0601, 6, 1;
    c0602, on_c0602, 6, 2;
    c0603, on_c0603, 6, 3;
    c0604, on_c0604, 6, 4;
    c0605, on_c0605, 6, 5;
    c0606, on_c0606, 6, 6;
    c0607, on_c0607, 6, 7;
    c0608, on_c0608, 6, 8;
    c0609, on_c0609, 6, 9;
    c0610, on_c0610, 6, 10;
    c0611, on_c0611, 6, 11;
    c0612, on_c0612, 6, 12;
    c0613, on_c0613, 6, 13;
    c0700, on_c0700, 7, 0;
    c0701, on_c0701, 7, 1;
    c0702, on_c0702, 7, 2;
    c0703, on_c0703, 7, 3;
    c0704, on_c0704, 7, 4;
    c0705, on_c0705, 7, 5;
    c0706, on_c0706, 7, 6;
    c0707, on_c0707, 7, 7;
    c0708, on_c0708, 7, 8;
    c0709, on_c0709, 7, 9;
    c0710, on_c0710, 7, 10;
    c0711, on_c0711, 7, 11;
    c0712, on_c0712, 7, 12;
    c0713, on_c0713, 7, 13;

}

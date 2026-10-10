use std::collections::HashMap;

use pchan_emu::debug::BreakpointKind;
use pchan_utils::hex;
use steel::steel_vm::builtin::BuiltInModule;
use steel::steel_vm::register_fn::RegisterFn;
use steel::{SteelErr, SteelVal};

#[derive(Clone, derive_more::Debug, PartialEq, Eq)]
#[debug("{}", hex(self.0))]
pub struct SteelU32(pub u32);

impl steel::rvals::Custom for SteelU32 {
    fn fmt(&self) -> Option<core::result::Result<String, core::fmt::Error>> {
        Some(Ok(format!("{self:?}")))
    }

    fn into_serializable_steelval(&mut self) -> Option<steel::rvals::SerializableSteelVal> {
        Some(steel::rvals::SerializableSteelVal::IntV(self.0 as isize))
    }

    fn visit_equality(&self, _visitor: &mut steel::rvals::cycles::EqualityVisitor) {}

    fn equality_hint(&self, other: &dyn steel::rvals::CustomType) -> bool {
        other
            .as_any_ref()
            .downcast_ref::<SteelU32>()
            .is_some_and(|other| self == other)
    }

    fn equality_hint_general(&self, other: &SteelVal) -> bool {
        matches!(other, SteelVal::IntV(num) if *num as u32 == self.0)
    }
}

impl SteelU32 {
    pub fn register(module: &mut BuiltInModule) {
        module.register_type::<SteelU32>("u32");
        module.register_fn("u32->int", |val: SteelU32| SteelVal::IntV(val.0 as isize));
    }
}

#[derive(Clone, derive_more::Debug)]
pub struct SteelGprMap(HashMap<&'static str, SteelU32>);

impl steel::rvals::Custom for SteelGprMap {
    fn fmt(&self) -> Option<core::result::Result<String, core::fmt::Error>> {
        Some(Ok(format!("{:#?}", self.0)))
    }
}

pub fn parse_rwx(value: &str) -> Result<BreakpointKind, SteelErr> {
    value
        .chars()
        .try_fold(BreakpointKind::NONE, |kind, c| match c {
            'r' => Ok(kind | BreakpointKind::READ),
            'w' => Ok(kind | BreakpointKind::WRITE),
            'x' => Ok(kind | BreakpointKind::EXECUTE),
            c => Err(SteelErr::new(
                steel::rerrs::ErrorKind::Parse,
                format!("Unexpected breakpoint kind: {c}"),
            )),
        })
}

use crate::{
    cranelift_bs::*,
    dynarec::{SummarizeDeps, SummarizeJit},
};
use std::fmt::Write;

use bon::Builder;

use pchan_utils::hex;
use petgraph::prelude::*;

#[derive(derive_more::Debug, Builder, Clone, derive_more::Display)]
#[display("{:#?}", self)]
pub struct JitSummary {
    #[debug("{}", self.decoded_ops)]
    pub decoded_ops: String,
    pub function: Option<Function>,
    pub panicked: bool,
    #[debug("{}", self.function_panic)]
    pub function_panic: String,
    #[debug("{}", self.cpu_state)]
    pub cpu_state: String,
}

impl SummarizeJit for JitSummary {
    fn summarize(deps: SummarizeDeps) -> Self {
        let cpu_state = format!("{:#?}", deps.cpu);
        let blocks = if let Some(fetch) = deps.fetch_summary {
            let mut dfs = Dfs::new(&fetch.cfg, fetch.entry);
            let mut buf = String::with_capacity(320);
            writeln!(&mut buf, "{{").unwrap();
            while let Some(node) = dfs.next(&fetch.cfg) {
                // tracing::trace!(cfg.node = ?fetch.cfg[node].clif_block);
                writeln!(&mut buf, "  {:?}:", fetch.cfg[node].clif_block()).unwrap();
                let ops = fetch.ops_for(&fetch.cfg[node]);
                let mut address = fetch.cfg[node].address;
                for op in ops {
                    writeln!(&mut buf, "    {}:    {op}", hex(address)).unwrap();
                    address += 4;
                }
                if ops.is_empty() {
                    writeln!(&mut buf, "    (empty)").unwrap();
                }
                _ = write!(&mut buf, "    => jumps to: ");
                let mut jumped = false;
                for n in fetch.cfg.neighbors_directed(node, Direction::Outgoing) {
                    _ = write!(&mut buf, "{:?}, ", fetch.cfg[n].clif_block());
                    jumped = true;
                }
                if !jumped {
                    _ = write!(&mut buf, "(none)");
                }
                _ = writeln!(&mut buf);
                writeln!(&mut buf).unwrap();
            }
            writeln!(&mut buf, "}}").unwrap();
            buf
        } else {
            "N/A (ops not passed to summarize)".to_string()
        };
        let panicked = deps
            .function_panic
            .as_ref()
            .map(|info| info.is_err())
            .unwrap_or(false);
        let panic = match deps.function_panic {
            Some(result) => match result {
                Ok(_) => "no panic. 👍".to_string(),
                Err(err) => format!("panic: {:?}", err),
            },
            None => "N/A (level `trace` required for this information.)".to_string(),
        };
        Self::builder()
            .maybe_function(deps.function.cloned())
            .function_panic(panic)
            .cpu_state(cpu_state)
            .decoded_ops(blocks)
            .panicked(panicked)
            .build()
    }
}

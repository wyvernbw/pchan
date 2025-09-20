use cranelift_codegen::ir;

use crate::dynarec::prelude::*;

/// Implementation of the [`InstBuilder`] that has
/// one convenience method per Cranelift IR instruction.
pub struct PureInstBuilder<'short, 'long: 'short> {
    pub builder: &'short mut FunctionBuilder<'long>,
    pub block: Block,
}

impl<'short, 'long> PureInstBuilder<'short, 'long> {
    fn new(builder: &'short mut FunctionBuilder<'long>, block: Block) -> Self {
        Self { builder, block }
    }
}

impl<'short, 'long> InstBuilderBase<'short> for PureInstBuilder<'short, 'long> {
    fn data_flow_graph(&self) -> &DataFlowGraph {
        &self.builder.func.dfg
    }

    fn data_flow_graph_mut(&mut self) -> &mut DataFlowGraph {
        &mut self.builder.func.dfg
    }

    fn build(self, data: InstructionData, ctrl_typevar: Type) -> (Inst, &'short mut DataFlowGraph) {
        // We only insert the Block in the layout when an instruction is added to it
        self.builder.ensure_inserted_block();

        let inst = self.builder.func.dfg.make_inst(data);
        self.builder.func.dfg.make_inst_results(inst, ctrl_typevar);

        (inst, &mut self.builder.func.dfg)
    }
}

"""Join independent attention query chunks on CPU; preserve all keys/queries/weights."""
import torch
def optimize_decoder(module):
    frozen=torch.jit.freeze(module.eval());graph=frozen.graph;joined=0
    for cat in list(graph.nodes()):
        if cat.kind()!='aten::cat' or cat.inputsAt(1).toIValue()!=-2:continue
        seq=cat.inputsAt(0).node()
        if seq.kind()!='prim::ListConstruct':continue
        values=list(seq.inputs())
        if not values or not all(v.node().kind()=='aten::scaled_dot_product_attention' for v in values):continue
        calls=[v.node() for v in values];unpack=calls[0].inputsAt(0).node()
        if unpack.kind()!='prim::ListUnpack':continue
        split=unpack.inputsAt(0).node()
        if split.kind()!='aten::split' or split.inputsAt(2).toIValue()!=-2:continue
        if [n.inputsAt(0).unique() for n in calls]!=[v.unique() for v in unpack.outputs()]:continue
        rest=list(calls[0].inputs())[1:]
        if any([v.unique() for v in list(n.inputs())[1:]]!=[v.unique() for v in rest] for n in calls):continue
        # Nonzero dropout, masks or causal position offsets would require separate analysis.
        if calls[0].inputsAt(3).toIValue() is not None or calls[0].inputsAt(4).toIValue()!=0.0 or calls[0].inputsAt(5).toIValue() is not False:continue
        node=graph.create('aten::scaled_dot_product_attention',[split.inputsAt(0),*rest],1)
        node.output().setType(cat.output().type());node.insertBefore(cat);cat.output().replaceAllUsesWith(node.output());joined+=1
    torch._C._jit_pass_dce(graph);graph.lint()
    if joined!=10:raise RuntimeError(f'Unexpected decoder attention layout: {joined} joins; retain original decoder')
    return frozen,{'joined_layers':joined,'attention_calls':sum(n.kind()=='aten::scaled_dot_product_attention' for n in graph.nodes())}

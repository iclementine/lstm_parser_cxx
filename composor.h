#pragma once

#include <dynet/dynet.h>
#include <dynet/model.h>
#include <dynet/expr.h>

namespace Utils{
	
using namespace dynet;

struct Composor {
public:
	Parameter p_wg;
	Parameter p_bg;
	Parameter p_wa;
	Parameter p_ba;
	Parameter p_wb;
	Parameter p_bb;
	unsigned size;
	
	ComputationGraph* pcg;
	Expression wg;
	Expression bg;
	Expression wa;
	Expression ba;
	Expression wb;
	Expression bb;
	
public:
	Composor(ParameterCollection& model, unsigned t_size):
			size(t_size) {
		p_wg = model.add_parameters({2 * size, 2 * size});
		p_bg = model.add_parameters({2 * size});
		p_wa = model.add_parameters({size, size});
		p_ba = model.add_parameters({size});
		p_wb = model.add_parameters({size, size});
		p_bb = model.add_parameters({size});
	}
	
	void new_graph(ComputationGraph& cg, bool update=true) {
		pcg = &cg;
		wg = update ? parameter(cg, p_wg) : const_parameter(cg, p_wg);
		bg = update ? parameter(cg, p_bg) : const_parameter(cg, p_bg);
		wa = update ? parameter(cg, p_wa) : const_parameter(cg, p_wa);
		ba = update ? parameter(cg, p_ba) : const_parameter(cg, p_ba);
		wb = update ? parameter(cg, p_wb) : const_parameter(cg, p_wb);
		bb = update ? parameter(cg, p_bb) : const_parameter(cg, p_bb);
}
	
	Expression operator()(const Expression& va, const Expression& vb) {
		Expression input = concatenate({va, vb});
		Expression gate = logistic(affine_transform({bg, wg, input}));
		Expression xa = cmult(pick_range(gate, 0, size), affine_transform({ba, wa, va}));
		Expression xb = cmult(pick_range(gate, size, 2 * size), affine_transform({bb, wb, vb}));
		Expression composed = xa + xb;
		return composed;
	}
};

} // namespace

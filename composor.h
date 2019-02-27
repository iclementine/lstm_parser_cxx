#pragma once

#include <dynet/dynet.h>
#include <dynet/model.h>
#include <dynet/expr.h>

namespace Utils{
	
using namespace dynet;

struct Composor {
public:
	Parameter p_w;
	Parameter p_b;
	unsigned size;
	ComputationGraph* pcg; Expression w; Expression b;
	
public:
	Composor(ParameterCollection& model, unsigned t_size):
			size(t_size) {
		p_w = model.add_parameters({2 * size, 2 * size});
		p_b = model.add_parameters({2 * size});
	}
	
	void new_graph(ComputationGraph& cg, bool update=true) {
		pcg = &cg;
		w = update ? parameter(cg, p_w) : const_parameter(cg, p_w);
		b = update ? parameter(cg, p_b) : const_parameter(cg, p_b);
}
	
	Expression operator()(const Expression& va, const Expression& vb) {
		Expression input = concatenate({va, vb});
		Expression gate = logistic(affine_transform({b, w, input}));
		Expression c = cmult(gate, input);
		Expression composed = pick_range(c, 0, size) + pick_range(c, size, 2 * size);
		return composed;
	}
};

} // namespace

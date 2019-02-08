#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>

#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <dynet/dynet.h>
#include <dynet/model.h>
#include <dynet/expr.h>
#include <dynet/lstm.h>
#include <dynet/training.h>
#include <dynet/io.h>

#include "corpus.h"

namespace po = boost::program_options;
using namespace std;
using namespace dynet;
using namespace treebank;
using namespace boost::posix_time;
using namespace boost::gregorian;

void InitCommandLine(int argc, char** argv, po::variables_map& conf) {
	using namespace std;
  // very typical function that uses side-effect, output to output parameter and returns nothing
	po::options_description opts("Configuration options");
	opts.add_options()
		("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
		("dev_data,d", po::value<string>(), "Development corpus")
		("test_data,p", po::value<string>(), "Test corpus")
		("epoches,e", po::value<unsigned>()->default_value(20), "Training epoches")
		("init_learning_rate,l", po::value<double>()->default_value(0.2), "Init learning rate")
		("resume,r", "Whether to resume training")
		("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
		("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
		("dropout_prob,D", po::value<double>()->default_value(0.3), "Dropout probability of lstms")
		("model,m", po::value<string>(), "Load saved model from this file")
		("use_pos_tags,P", "make POS tags visible to parser")
		("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
		("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
		("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
		("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
		("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
		("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
		("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
		("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
		("train,t", "Should training be run?")
		("words,w", po::value<string>(), "Pretrained word embeddings")
		("help,h", "Help");
	po::options_description dcmdline_options;
	dcmdline_options.add(opts);
	po::store(parse_command_line(argc, argv, dcmdline_options), conf);
	po::notify(conf);
	if (conf.count("help")) {
		cerr << dcmdline_options << endl;
		exit(1);
	}
	if (conf.count("training_data") == 0) {
		cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
		exit(1);
	}
}

struct ParserBuilder{
	
public:
	bool use_pos;
	bool use_pretrained;
	double p_unk;
	double p_dropout;
	LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
	LSTMBuilder buffer_lstm;
	LSTMBuilder action_lstm;
	LookupParameter p_w; // word embeddings
	LookupParameter p_t; // pretrained word embeddings (not updated)
	LookupParameter p_a; // input action embeddings
	LookupParameter p_r; // relation embeddings
	LookupParameter p_p; // pos tag embeddings
	Parameter p_pbias; // parser state bias
	Parameter p_A; // action lstm to parser state
	Parameter p_B; // buffer lstm to parser state
	Parameter p_S; // stack lstm to parser state
	Parameter p_H; // head matrix for composition function
	Parameter p_D; // dependency matrix for composition function
	Parameter p_R; // relation matrix for composition function
	Parameter p_w2l; // word to LSTM input
	Parameter p_p2l; // POS to LSTM input
	Parameter p_t2l; // pretrained word embeddings to LSTM input
	Parameter p_ib; // LSTM input bias
	Parameter p_cbias; // composition function bias
	Parameter p_p2a;   // parser state to action
	Parameter p_action_start;  // action bias
	Parameter p_abias;  // action bias
	Parameter p_buffer_guard;  // end of buffer
	Parameter p_stack_guard;  // end of stack
	ParameterCollection* pc;  //
  
public:
	explicit ParserBuilder(ParameterCollection& model, unsigned layers, unsigned lstm_input_dim,
												 unsigned action_dim, unsigned action_size, unsigned input_dim, unsigned vocab_size, 
												 double t_p_unk, double t_p_dropout, unsigned rel_dim, unsigned hidden_dim,
												 bool t_use_pretrained, const unordered_map<unsigned, vector<float>>& pretrained,
												 bool t_use_pos, unsigned pos_size, unsigned pos_dim) :
			use_pos(t_use_pos), use_pretrained(t_use_pretrained), 
			pc(&model), p_unk(t_p_unk),p_dropout(t_p_dropout),
			stack_lstm(layers, lstm_input_dim, hidden_dim, model),
			buffer_lstm(layers, lstm_input_dim, hidden_dim, model),
			action_lstm(layers, action_dim, hidden_dim, model) {
		p_w = model.add_lookup_parameters(vocab_size, {input_dim});
		p_a = model.add_lookup_parameters(action_size, {action_dim});
		p_r = model.add_lookup_parameters(action_size, {rel_dim});
		p_pbias = model.add_parameters({hidden_dim});
		p_A = model.add_parameters({hidden_dim, hidden_dim});
		p_B = model.add_parameters({hidden_dim, hidden_dim});
		p_S = model.add_parameters({hidden_dim, hidden_dim});
		p_H = model.add_parameters({lstm_input_dim, lstm_input_dim});
		p_D = model.add_parameters({lstm_input_dim, lstm_input_dim});
		p_R = model.add_parameters({lstm_input_dim, rel_dim});
		p_w2l = model.add_parameters({lstm_input_dim, input_dim});
		p_ib = model.add_parameters({lstm_input_dim});
		p_cbias = model.add_parameters({lstm_input_dim});
		p_p2a= model.add_parameters({action_size, hidden_dim});
		p_action_start = model.add_parameters({action_dim});
		p_abias = model.add_parameters({action_size});
		p_buffer_guard = model.add_parameters({lstm_input_dim});
		p_stack_guard = model.add_parameters({lstm_input_dim});
		if (use_pos) {
			p_p = model.add_lookup_parameters(pos_size, {pos_dim});
			p_p2l = model.add_parameters({lstm_input_dim, pos_dim});
		}
		if (use_pretrained) {
			unsigned pretrained_dim = pretrained.at(0).size();
			p_t = model.add_lookup_parameters(pretrained.size(), {pretrained_dim});
			for (auto it: pretrained)
				p_t.initialize(it.first, it.second);
			p_t2l = model.add_parameters({lstm_input_dim, pretrained_dim});
		}
	}

	// LEGAL FUNCTION
	static bool IsActionFeasible(const string& a, unsigned bsize, unsigned ssize, const vector<int>& stacki) {
		if (a[1]=='w') {
			if ((ssize > 3) && (stacki[stacki.size()-2] < stacki[stacki.size()-1])) return true;
		} else if (a[0] == 'S' && a[1]=='h') {
			if (bsize > 1) return true;
		} else if (a[0] == 'L') {
			if (ssize > 3) return true;
		} else if (a[0] == 'R') {
			if ((ssize > 3) || (ssize == 3) && (bsize == 1)) return true;
		}
		return false;
	}
	
	// take a vector of actions and return a parse tree (labeling of every
	// word position with its head's position) heads & rels, inclu that for <root>
	static tuple<vector<int>, vector<string>> compute_heads(unsigned sent_len, const vector<unsigned>& actions, 
																														const Vocab& transition) {
		vector<int> heads(sent_len, -1);
		vector<string> rels(sent_len, "");

		vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
		for (unsigned i = 0; i < sent_len; ++i)
			bufferi[sent_len - i] = i;
		bufferi[0] = -999;
		stacki.push_back(bufferi.back()); bufferi.pop_back();
		
		for (auto action: actions) { // loop over transitions for sentence
			const string& actionString=transition.itos.at(action);
			const char ac = actionString[0];
			const char ac2 = actionString[1];
			if (ac =='S' && ac2=='h') {  // SHIFT
				assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
				stacki.push_back(bufferi.back());
				bufferi.pop_back();
			} else if (ac=='S' && ac2=='w') { // SWAP
				assert(stacki.size() > 3);
				unsigned ii = 0, jj = 0;
				jj = stacki.back();
				stacki.pop_back();
				ii = stacki.back();
				stacki.pop_back();
				bufferi.push_back(ii);
				stacki.push_back(jj);
			} else { // LEFT or RIGHT
				assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
				assert(ac == 'L' || ac == 'R');
				
				size_t first_char_in_rel = actionString.find('-') + 1; 
				string hyp_rel = actionString.substr(first_char_in_rel);
				unsigned depi = 0, headi = 0;
				(ac == 'R' ? depi : headi) = stacki.back();
				stacki.pop_back();
				(ac == 'R' ? headi : depi) = stacki.back();
				stacki.pop_back();
				stacki.push_back(headi);
				heads[depi] = headi;
				rels[depi] = hyp_rel;
			}
		}
		assert(bufferi.size() == 1);
		assert(stacki.size() == 2);
		return make_tuple(heads, rels);
	}

	static tuple<unsigned, unsigned> compute_correct(const vector<int>& ref, const vector<int>& hyp, 
																									 const vector<string> rel_ref, const vector<string> rel_hyp,
																									 unsigned len) {
		unsigned correct_head = 0, correct_rel = 0;
		for (unsigned i = 1; i < len; ++i) {
			if (hyp[i] == ref[i]) {
				++ correct_head;
				if (rel_hyp[i] == rel_ref[i]) ++correct_rel;
			}
		}
		return make_tuple(correct_head, correct_rel);
	}
	
	void output_conll(const vector<string>& sent, const vector<string>& sent_pos,
										const vector<int>& hyp, const vector<string>& rel_hyp) {
		for (unsigned i = 1; i < sent.size(); ++i) {
			string wit = sent[i];
			string pit = sent_pos[i];
			int hyp_head = hyp[i];
			string hyp_rel = rel_hyp[i];
			cout << i << '\t'       // 1. ID 
					<< wit << '\t'         // 2. FORM
					<< "_" << '\t'         // 3. LEMMA 
					<< "_" << '\t'         // 4. CPOSTAG 
					<< pit << '\t' // 5. POSTAG
					<< "_" << '\t'         // 6. FEATS
					<< hyp_head << '\t'    // 7. HEAD
					<< hyp_rel << '\t'     // 8. DEPREL
					<< "_" << '\t'         // 9. PHEAD
					<< "_" << endl;        // 10. PDEPREL
		}
		cout << endl;
	}

	tuple<vector<unsigned>, Expression> log_prob_parser(ComputationGraph& hg, const vector<unsigned>& sent, 
																											const vector<unsigned>& sent_pos, const vector<unsigned> correct_actions,
																											const Vocab& transition, const Vocab& form, double* right) {
		vector<unsigned> results;
		const bool build_training_graph = correct_actions.size() > 0;
		// set dropout 
		if (build_training_graph) {
			stack_lstm.set_dropout(0.3); buffer_lstm.set_dropout(0.3); action_lstm.set_dropout(0.3);
		} else {
			stack_lstm.disable_dropout(); buffer_lstm.disable_dropout(); action_lstm.disable_dropout();
		}
		stack_lstm.new_graph(hg);
		buffer_lstm.new_graph(hg);
		action_lstm.new_graph(hg);
		stack_lstm.start_new_sequence();
		buffer_lstm.start_new_sequence();
		action_lstm.start_new_sequence();
		// variables in the computation graph representing the parameters
		Expression pbias = parameter(hg, p_pbias);
		Expression H = parameter(hg, p_H);
		Expression D = parameter(hg, p_D);
		Expression R = parameter(hg, p_R);
		Expression cbias = parameter(hg, p_cbias);
		Expression S = parameter(hg, p_S);
		Expression B = parameter(hg, p_B);
		Expression A = parameter(hg, p_A);
		Expression ib = parameter(hg, p_ib);
		Expression w2l = parameter(hg, p_w2l);
		Expression p2l;
		if (use_pos)
			p2l = parameter(hg, p_p2l);
		Expression t2l;
		if (use_pretrained)
			t2l = parameter(hg, p_t2l);
		Expression p2a = parameter(hg, p_p2a);
		Expression abias = parameter(hg, p_abias);
		Expression action_start = parameter(hg, p_action_start);

		action_lstm.add_input(action_start);

		vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
		vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
		// precompute buffer representation from left to right

		for (unsigned i = 0; i < sent.size(); ++i) {
			assert(sent[i] < form.stoi.size());
			Expression w =lookup(hg, p_w, sent[i]);

			vector<Expression> args = {ib, w2l, w}; // learn embeddings
			if (use_pos) { // learn POS tag?
				Expression p = lookup(hg, p_p, sent_pos[i]);
				args.push_back(p2l);
				args.push_back(p);
			}
			if (use_pretrained) {  // include fixed pretrained vectors?
				Expression t = const_lookup(hg, p_t, sent[i]);
				args.push_back(t2l);
				args.push_back(t);
			}
			buffer[sent.size() - i] = rectify(affine_transform(args)); // reversed
			bufferi[sent.size() - i] = i;
		}
		// dummy symbol to represent the empty buffer
		buffer[0] = parameter(hg, p_buffer_guard);
		bufferi[0] = -999;
		for (auto& b : buffer)
			buffer_lstm.add_input(b);

		vector<Expression> stack;  // variables representing subtree embeddings
		vector<int> stacki; // position of words in the sentence of head of subtree
		stack.push_back(parameter(hg, p_stack_guard));
		stacki.push_back(-999); // not used for anything
		// drive dummy symbol on stack through LSTM
		stack_lstm.add_input(stack.back());
		
		// push root to the stack
		stack.push_back(buffer.back());
		stack_lstm.add_input(buffer.back());
		buffer.pop_back();
		buffer_lstm.rewind_one_step();
		stacki.push_back(bufferi.back());
		bufferi.pop_back();
		
		/* debug
		cout << "stacki: " << stacki << endl;
		cout << "bufferi: " << bufferi << endl;
		*/
		
		// End of "Setup" code, now start the main loop
		vector<Expression> log_probs;
		string rootword; //
		unsigned action_count = 0;  // incremented at each prediction
		while(stack.size() > 2 || buffer.size() > 1) {
			// get list of possible actions for the current parser state
			vector<unsigned> current_valid_actions;
			for (const auto& action: transition.stoi)
				if (IsActionFeasible(action.first, buffer.size(), stack.size(), stacki))
					current_valid_actions.push_back(action.second);

			// p_t = pbias + S * slstm + B * blstm + A * almst
			Expression p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
			Expression nlp_t = rectify(p_t);
			// r_t = abias + p2a * nlp
			Expression r_t = affine_transform({abias, p2a, nlp_t});

			// adist = log_softmax(r_t, current_valid_actions)
			Expression adiste = log_softmax(r_t, current_valid_actions);
			vector<float> adist = as_vector(hg.incremental_forward(adiste));
			double best_score = adist[current_valid_actions[0]];
			unsigned best_a = current_valid_actions[0];
			for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
				if (adist[current_valid_actions[i]] > best_score) {
					best_score = adist[current_valid_actions[i]];
					best_a = current_valid_actions[i];
				}
			}
			unsigned action = best_a;
			if (build_training_graph) {  // if we have reference actions (for training) use the reference action
				action = correct_actions[action_count];
				if (best_a == action) { (*right)++; }
			}
			++action_count;
			log_probs.push_back(pick(adiste, action));
			results.push_back(action);

			// add current action to action LSTM
			Expression actione = lookup(hg, p_a, action);
			action_lstm.add_input(actione);

			// get relation embedding from action (TODO: convert to relation from action?)
			Expression relation = lookup(hg, p_r, action);

			// do action
			const string& actionString=transition.itos.at(action);
			const char ac = actionString[0];
			const char ac2 = actionString[1];


			if (ac =='S' && ac2=='h') {  // SHIFT
				assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
				stack.push_back(buffer.back());
				stack_lstm.add_input(buffer.back());
				buffer.pop_back();
				buffer_lstm.rewind_one_step();
				stacki.push_back(bufferi.back());
				bufferi.pop_back();
			} else if (ac=='S' && ac2=='w'){ //SWAP --- Miguel
				assert(stack.size() > 3); // dummy symbol and root symbol means > 3 (not >= 3)

				Expression toki, tokj;
				unsigned ii = 0, jj = 0;
				tokj=stack.back();
				jj=stacki.back();
				stack.pop_back();
				stacki.pop_back();

				toki=stack.back();
				ii=stacki.back();
				stack.pop_back();
				stacki.pop_back();

				buffer.push_back(toki);
				bufferi.push_back(ii);

				stack_lstm.rewind_one_step();
				stack_lstm.rewind_one_step();

				buffer_lstm.add_input(buffer.back());

				stack.push_back(tokj);
				stacki.push_back(jj);

				stack_lstm.add_input(stack.back());
			} else { // LEFT or RIGHT
				assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
				assert(ac == 'L' || ac == 'R');
				Expression dep, head;
				unsigned depi = 0, headi = 0;
				(ac == 'R' ? dep : head) = stack.back();
				(ac == 'R' ? depi : headi) = stacki.back();
				stack.pop_back();
				stacki.pop_back();
				(ac == 'R' ? head : dep) = stack.back();
				(ac == 'R' ? headi : depi) = stacki.back();
				stack.pop_back();
				stacki.pop_back();
				if (headi == 0) rootword = form.itos.at(sent[depi]);
				// composed = cbias + H * head + D * dep + R * relation
				Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
				Expression nlcomposed = tanh(composed);
				stack_lstm.rewind_one_step();
				stack_lstm.rewind_one_step();
				stack_lstm.add_input(nlcomposed);
				stack.push_back(nlcomposed);
				stacki.push_back(headi);
			}
		}
		assert(stack.size() == 2); // guard symbol, root
		assert(stacki.size() == 2);
		assert(buffer.size() == 1); // guard symbol
		assert(bufferi.size() == 1);
		Expression tot_neglogprob = -sum(log_probs);
		assert(tot_neglogprob.pg != nullptr);
		return make_tuple(results, tot_neglogprob);
	}

	void train(const vector<Sentence>& train_sentences, const vector<Sentence>& dev_sentences, 
						 const vector<Sentence>& test_sentences, const Vocab& form, const Vocab& transition,
						 unsigned epoch, double lr, unsigned status_every_i_iterations,
						 bool resume, string param, string train_state) {
		// load parameter collection
		if (resume && param.size()) {
			TextFileLoader l(param);
			l.populate(*pc);
		}
		
		// load trainer state
		SimpleSGDTrainer trainer(*pc, lr);
		if (resume && train_state.size()) {
			ifstream is(train_state);
			trainer.populate(is);
		}
		
		// stat
		unsigned trs = 0; double right = 0; double llh; //clear every status for output
		int iter = 0; unsigned tot_seen = 0; // always incremented
		unsigned sid = 0; // sentence id, clear every epoch
		double best_uas = 0, best_las = 0;
		
		// ids
		vector<unsigned> ids = vector<unsigned>(train_sentences.size(), 0);
		for (unsigned i = 0; i < train_sentences.size(); ++i) ids[i] = i;
		
		cout << "=====training begin=====" << endl;
		while (tot_seen / train_sentences.size() < epoch) {
			if (sid == train_sentences.size()) {
				double uas, las;
				tie(uas, las) = test(test_sentences, form, transition, status_every_i_iterations, false, string(), false);
				if (uas > best_uas) {
					TextFileSaver s(param); s.save(*pc);
				}
				random_shuffle(ids.begin(), ids.end()); sid = 0; trainer.learning_rate *= 0.9;
			}
			
			if ((tot_seen > 0) && (tot_seen % status_every_i_iterations == 0)) {
				trainer.status();
				ptime time_now{second_clock::local_time()};
      	cout << "update #" << iter << " (epoch " << (double(tot_seen) / train_sentences.size()) 
					<< " |time=" << time_now << ")\tllh: " << llh 
					<<" ppl: " << exp(llh / trs) 
					<< " err: " << (trs - right) / trs << endl;
				llh = trs = right = 0;
			}

			ComputationGraph hg;
			const Sentence& sent = train_sentences[ids[sid]];
			vector<unsigned> results; Expression losse;
			tie(results, losse)= log_prob_parser(hg, sent.formi, sent.posi, sent.transitionsi, transition, form, &right);
			double loss = as_scalar(hg.forward(losse));
			hg.backward(losse); trainer.update();
			++sid; ++tot_seen; ++iter; trs += sent.transitions.size(); llh += loss;
		}
		
		// final test on test Setup
		test(test_sentences, form, transition, status_every_i_iterations, true, param, true);
	}
	
	tuple<double, double> test(const vector<Sentence>& test_sentences, const Vocab& form, const Vocab& transition,
						unsigned status_every_i_iterations, bool resume, string param, bool output) {
		// load parameter collection
		if (resume && param.size()) {
			TextFileLoader l(param);
			l.populate(*pc);
		}
		
		//stat
		ptime start = microsec_clock::local_time();
		unsigned tot_tokens = 0; unsigned correct_head = 0; unsigned correct_rel = 0;
		double right = 0;
		for (unsigned i = 0; i < test_sentences.size(); ++i) {
			ComputationGraph hg;
			if ((!output) && (i % 200 == 0))
				cout << i << "/" << test_sentences.size() << endl;
			const Sentence& sent = test_sentences[i];
			vector<unsigned> results; Expression losse;
			tie(results, losse) = log_prob_parser(hg, sent.formi, sent.posi, sent.transitionsi, transition, form, &right);
			double loss = as_scalar(hg.forward(losse));
			// compute heads and 
			vector<int> hyp; vector<string> rels_hyp;
			tie(hyp, rels_hyp) = compute_heads(sent.form.size(), results, transition);
			unsigned h = 0, r = 0;
			tie(h, r) = compute_correct(sent.head, hyp, sent.deprel, rels_hyp, sent.form.size());
			correct_head += h; correct_rel += r; tot_tokens += sent.form.size() - 1;
			if (output)
				output_conll(sent.form, sent.pos, hyp, rels_hyp);
		}
		ptime end = microsec_clock::local_time();
		if (!output) 
			cout << "test " << test_sentences.size() << " sentences in " << (end - start).total_milliseconds() << " ms." << endl;
		double uas = float(correct_head) / tot_tokens;
		double las = float(correct_rel) / tot_tokens;
		cout << "uas: " << uas << "\tlas: " << las << endl;
		return make_tuple(uas, las);
	}
};

int main(int argc, char** argv) {
	
	// parse command line args and prepare args for ParserBuilder
	po::variables_map conf;
	InitCommandLine(argc, argv, conf);
	
	auto training_data = conf["training_data"].as<string>();
	auto dev_data = conf["dev_data"].as<string>();
	auto test_data = conf["test_data"].as<string>();
	
	// Read in corpus
	cout << "Reading corpus..." << endl;
	Corpus corpus;
	corpus.load_corpus(corpus.train_sentences, training_data);
	corpus.load_corpus_dev(corpus.dev_sentences, dev_data);
	corpus.load_corpus_dev(corpus.test_sentences, test_data);
// 	cout << corpus.form << endl << corpus.pos << endl 
// 	<< corpus.deprel << endl << corpus.transition << endl
// 	<< corpus.chars << endl;
	
	const auto epoch = conf["epoches"].as<unsigned>();
	const auto lr = conf["init_learning_rate"].as<double>();
	bool resume = conf.count("resume");
	const auto layers = conf["layers"].as<unsigned>();
	const auto lstm_input_dim = conf["lstm_input_dim"].as<unsigned>();
	const auto action_dim = conf["action_dim"].as<unsigned>();
	const auto action_size = corpus.transition.stoi.size();
	const auto input_dim = conf["input_dim"].as<unsigned>();
	const auto vocab_size = corpus.form.stoi.size();
	const auto rel_dim = conf["rel_dim"].as<unsigned>();
	const auto hidden_dim = conf["hidden_dim"].as<unsigned>();
	const auto unk_strategy = conf["unk_strategy"].as<unsigned>();
	const auto p_unk = conf["unk_prob"].as<double>();
	const auto p_dropout = conf["dropout_prob"].as<double>();

	// read in pretrained word embedding, I love iostream, stringstream, string, so good
	unordered_map<unsigned, vector<float>> pretrained;
	bool use_pretrained = conf.count("words");
	if (use_pretrained) {
		auto pretrained_dim = conf["pretrained_dim"].as<unsigned>();
		const unsigned unk_id = corpus.form.get_or_add(Vocab::UNK);
		unsigned max_id = corpus.form.stoi.size();
		pretrained[unk_id] = vector<float>(pretrained_dim, 0);
		cerr << "Loading from " << conf["words"].as<string>() << " with" << pretrained_dim << " dimensions\n";
		ifstream in(conf["words"].as<string>());
		string line;
		getline(in, line);
		vector<float> v(pretrained_dim, 0);
		string word;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> word;
			for (unsigned i = 0; i < pretrained_dim; ++i) lin >> v[i];
			unsigned id = corpus.form.get_or_add(word);
			if (id != unk_id)
				pretrained[id] = v;
			else
				pretrained[max_id++] = v; // pretrained has possibly more word entries than corpus.form.itos
		}
	}
	
	bool use_pos = conf.count("use_pos_tags");
	unsigned pos_size = 0; 
	unsigned pos_dim;
	if (use_pos) {
		pos_size = corpus.pos.stoi.size();
		pos_dim = conf["pos_dim"].as<unsigned>();
	}
	const unsigned status_every_i_iterations = 100;
	
	// print conf 
	cout << "======Configuration is=====" << endl
		<< "training_data: " << training_data << endl
		<< "training epoches: " << epoch << endl
		<< "init learning rate: " << lr << endl 
		<< "dev_data: " << dev_data << endl
		<< "layers: " << layers << endl
		<< "lstm_input_dim: " << lstm_input_dim << endl
		<< "dropout_prob: " << p_dropout << endl
		<< "action_dim: " << action_dim << endl
		<< "action_size: " << action_size << endl
		<< "input_dim: " << input_dim << endl
		<< "vocab_size: " << vocab_size << endl
		<< "rel_dim: " << rel_dim << endl
		<< "hidden_dim: "  << hidden_dim << endl
		<< "use_pos_tags: " << use_pos << endl;
	if (use_pos) 
		cout << "pos_dim: " << pos_dim << endl
			<< "pos_size: " << pos_size << endl;
	cout << "use_pretrained: " << use_pretrained << endl;
	if (use_pretrained) 
		cout << "pretrained embeddings size is " << pretrained.size() << endl;
	
  cout << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cout << "stochastic replacement of hapax legomena\n";
		cout << "unk_prob: " << p_unk << endl;
  } else {
    abort();
  }
	cout << "====================" << endl;
	
	// dynet build model
	dynet::initialize(argc, argv, false);
	ParameterCollection model;
	ParserBuilder parser(model, layers, lstm_input_dim, action_dim, action_size, input_dim, vocab_size, p_unk, p_dropout,
											 rel_dim, hidden_dim, use_pretrained, pretrained, use_pos, pos_size, pos_dim);
	parser.train(corpus.train_sentences, corpus.dev_sentences, 
							 corpus.test_sentences, corpus.form, corpus.transition,
							 epoch, lr, status_every_i_iterations,
							 resume, "parser.model", "trainer.state");

	return 0;
}

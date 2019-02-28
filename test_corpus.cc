#include <iostream>
#include "corpus.h"

using namespace std;
using namespace treebank;

template <typename T>
ostream& operator<< (ostream& os, const vector<T>& vec) {
	for (const T& x: vec)
		cerr << x << ", ";
	return os;
}

int main(){
	cout << "hello world!" << endl;
	Corpus corpus;

	corpus.load_corpus(corpus.train_sentences, "../experiment/arc-hybrid/train-arc-hybrid.txt");
	corpus.load_corpus_dev(corpus.dev_sentences, "../experiment/arc-hybrid/dev-arc-hybrid.txt");
	corpus.load_corpus_dev(corpus.test_sentences, "../experiment/arc-hybrid/test-arc-hybrid.txt");
	cout << corpus.form << endl << corpus.pos << endl 
		<< corpus.deprel << endl << corpus.transition << endl
		<< corpus.chars << endl;
	
	cout << "There are " << corpus.train_sentences.size() << " train sentences in the corpus" << endl;
	cout << "There are " << corpus.dev_sentences.size() << " dev sentences in the corpus" << endl;
	cout << "There are " << corpus.test_sentences.size() << " test sentences in the corpus" << endl;
	cout << corpus.train_sentences[0].form << endl;
	return 0;
}


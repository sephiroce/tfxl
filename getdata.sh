mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
if [[ ! -d 'ptb' ]]; then
    wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    tar -xzf simple-examples.tgz

    mkdir -p ptb
    cd ptb
    mv ../simple-examples/data/ptb.train.txt train.txt
    mv ../simple-examples/data/ptb.test.txt test.txt
    mv ../simple-examples/data/ptb.valid.txt valid.txt
    cd ..

    rm -rf simple-examples/
fi

echo "- Downloading 1B words"

if [[ ! -d 'one-billion-words' ]]; then
    mkdir -p one-billion-words
    cd one-billion-words

    wget --no-proxy http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
    tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

    path="1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
    cat ${path}/news.en.heldout-00000-of-00050 > valid.txt
    cat ${path}/news.en.heldout-00000-of-00050 > test.txt

    wget https://github.com/rafaljozefowicz/lm/raw/master/1b_word_vocab.txt

    cd ..
fi


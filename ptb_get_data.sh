mkdir -p data
cd data

if [[ ! -d 'ptb' ]]; then
  echo "Downloading & preprocessing Penn Treebank (PTB)"
  wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  tar -xzf simple-examples.tgz

  mkdir -p ptb
  cd ptb
  mv ../simple-examples/data/ptb.train.txt train.txt
  mkdir train
  cd train
  split -l 20000 ../train.txt train.
  cd ..
  rm train.txt
  mv ../simple-examples/data/ptb.test.txt test.txt
  mv ../simple-examples/data/ptb.valid.txt valid.txt

  echo "Building up a vocabulary file of PTB"
  sed -e 's/ /\n/g' train/train.?? valid.txt test.txt | sort -u > ptb.vocab
  echo "<s>" >> ptb.vocab
  echo "</s>" >> ptb.vocab
  sed -i '/^$/d' ptb.vocab
  vocab=`wc -l ptb.vocab`
  echo ptb.vocab consists of ${vocab} words.

  # Cleaning up
  cd ..
  rm -rf simple-examples/
  rm simple-examples.tgz
fi
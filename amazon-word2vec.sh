for dim in 100 150 200 250 350 400 450 500
do
    time make run-word2vec datafile=~/data/amazon/processed-reviews.txt size=${dim}
done

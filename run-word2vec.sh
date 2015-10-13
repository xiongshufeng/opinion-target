DIMS="10 30 50 70 90 110 130 150 170 190 210 230 250 500 900 1300"
for length in $DIMS
do
    time make run-word2vec datafile=enwik9.txt size=${length}
done

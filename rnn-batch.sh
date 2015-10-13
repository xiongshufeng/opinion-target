embedding=$1
mkdir -p rnn-experiments

echo "[INFO] ------------------Begin of Embedding: $embedding---------------"

if [ "$embedding" == "Google" ]; then
    file=embeddings/google-news/GoogleNews-vectors-negative300.bin.gz
    dim=300
elif [ "$embedding" == "Amazon" ]; then
    file=embeddings/amazon/vectors-$2.txt
    dim=$2
else
    file=embeddings/senna
    dim=50
fi

make laptop-json EMBEDDIING_FILE=$file embed=$embedding
make restaurant-json EMBEDDIING_FILE=$file embed=$embedding

for type in elman
do
    for units in 50 100 150 200
    do
        make run-rnn dataset=laptop type=${type} embed=${embedding} window=3 nhidden=${units} dimension=${dim} init=true > rnn-experiments/${embedding}-laptop-${type}-3-${units}-${dim}-true.txt &
        make run-rnn dataset=restaurant type=${type} embed=${embedding} window=3 nhidden=${units} dimension=${dim} init=true > rnn-experiments/${embedding}-restaurant-${type}-3-${units}-${dim}-true.txt &
    done
done

echo "[INFO] ------------------End of Embedding: $embedding---------------"

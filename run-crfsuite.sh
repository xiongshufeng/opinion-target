name=$1
embedding=$2
mkdir -p vector-experiments
echo "[INFO] ------------------Begin---------------"
if [ "$embedding" == "Google" ]; then
    file=embeddings/google-news/GoogleNews-vectors-negative300.bin.gz
elif [ "$embedding" == "Amazon" ]; then
    file=embeddings/amazon/vectors-$3.txt
else
    file=embeddings/senna
fi

echo "[INFO] ------------------Begin of Embedding: $embedding---------------"
make $name-features EMBEDDIING_FILE=$file type=$embedding
make run-crfsuite dataset=$name type=bin > vector-experiments/$name-bin-$embedding.txt
make run-crfsuite dataset=$name type=con > vector-experiments/$name-con-$embedding.txt
make run-crfsuite dataset=$name type=bc > vector-experiments/$name-bc-$embedding.txt

echo "[INFO] Binary features: $embedding "
tail -14 vector-experiments/$name-bin-$embedding.txt
echo "[INFO] Continuous features: $embedding"
tail -14 vector-experiments/$name-con-$embedding.txt
echo "[INFO] Binary-Continuous features: $embedding"
tail -14 vector-experiments/$name-bc-$embedding.txt
echo "[INFO] ------------------End of Embedding: $embedding---------------"

echo "[INFO] ------------------END---------------"

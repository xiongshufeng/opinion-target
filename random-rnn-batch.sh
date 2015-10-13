make laptop-json
make restaurant-json
for dim in 50 100 150 200 250 300
do
    for type in elman jordan
    do
        for units in 50 100 150 200 250 300
        do
            make run-rnn dataset=laptop type=${type} embed=Senna window=3 nhidden=${units} dimension=${dim} init=false > laptop-${type}-3-${units}-${dim}-false.txt
            make run-rnn dataset=restaurant type=${type} embed=Senna window=3 nhidden=${units} dimension=${dim} init=false > restaurant-${type}-3-${units}-${dim}-false.txt
        done
    done
done

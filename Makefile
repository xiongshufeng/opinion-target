###################################################################################################
## Experiments for Baseline
###################################################################################################
laptop-baseline:
	rm -f baseline/*.xml
	cd baseline; python semeval_base.py --train ../evaluation/Laptop_Train_v2.xml --task 1

restaurant-baseline:
	rm -f baseline/*.xml
	cd baseline; python semeval_base.py --train ../evaluation/Restaurants_Train_v2.xml --task 5

###################################################################################################
## Experiments for Conditional Random Field
###################################################################################################
prepare-dataset:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFfile -Dexec.args="evaluation/Restaurants_Train_v2.xml restaurant-train.tsv"
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFfile -Dexec.args="evaluation/Restaurants_Test_Data_PhaseB.xml restaurant-test.tsv"
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFfile -Dexec.args="evaluation/Laptop_Train_v2.xml laptop-train.tsv"	
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFfile -Dexec.args="evaluation/Laptops_Test_Data_PhaseB.xml laptop-test.tsv"	

evaluate-restaurant:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppOpinionTarget \
		-Dexec.args="restaurant-model.ser.gz restaurant-train.tsv evaluation/Restaurants_Test_Data_PhaseA.xml restaurant-test-predict.tsv"
	paste restaurant-test.tsv restaurant-test-predict.tsv | cut -f1,3,5 > restaurant-evaluate.tsv
	perl conlleval.pl -d "\t" < restaurant-evaluate.tsv

###################################################################################################
## Experiments for Recurrent Neural Network
###################################################################################################	
LAPTOP_TRAIN=evaluation/Laptop_Train_v2.xml
LAPTOP_TEST=evaluation/Laptops_Test_Data_PhaseB.xml

RESTAURANT_TRAIN=evaluation/Restaurants_Train_v2.xml
RESTAURANT_TEST=evaluation/Restaurants_Test_Data_PhaseB.xml

embed=Senna
ifeq (${embed}, Google)
	EMBEDDIING_FILE=embeddings/google-news/GoogleNews-vectors-negative300.bin.gz
else
	ifeq (${embed}, Amazon)
		EMBEDDIING_FILE=embeddings/amazon/vectors-300.txt
	else
		EMBEDDIING_FILE=embeddings/senna
	endif
endif

laptop-json:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppPrepareRnnDataset \
	-Dexec.args="-t ${LAPTOP_TRAIN} -r 0.9 -s ${LAPTOP_TEST} -o laptop-json-${embed}.txt -e ${EMBEDDIING_FILE} -p ${embed}"
	
restaurant-json:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppPrepareRnnDataset \
	-Dexec.args="-t ${RESTAURANT_TRAIN} -r 0.9 -s ${RESTAURANT_TEST} -o restaurant-json-${embed}.txt -e ${EMBEDDIING_FILE} -p ${embed}"

# Example: make run-rnn dataset=laptop embed=Senna type=elman window=3 nhidden=100 dimension=50 init=true
run-rnn:
	python main.py ${dataset}-json-${embed}.txt ${type} ${dataset}-${type}-${embed}-${window}-${nhidden}-${dimension} ${window} ${nhidden} ${dimension} ${init}

###################################################################################################
## Experiments for CRFsuite on Word Embeddings
###################################################################################################	
run-word2vec:
	cd word2vec; ./word2vec -train ${datafile} -output vectors-${size}.txt -cbow 0 -size ${size} -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 0
	@echo "[INFO] word2vec is finished."

laptop-features:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFVectorFile -Dexec.args="${LAPTOP_TRAIN} ${EMBEDDIING_FILE} ${embed} laptop-train.tsv laptop-train.con"
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFVectorFile -Dexec.args="${LAPTOP_TEST} ${EMBEDDIING_FILE} ${embed} laptop-test.tsv laptop-test.con"
	cat laptop-train.tsv | ./absa.py -s '\t' > laptop-train.bin
	cat laptop-test.tsv | ./absa.py -s '\t' > laptop-test.bin
	cut -f 2- laptop-train.bin > laptop-train-bool.tmp
	cut -f 2- laptop-test.bin > laptop-test-bool.tmp
	paste laptop-train.con laptop-train-bool.tmp | sed 's/^\t$///g' > laptop-train.bc
	paste laptop-test.con laptop-test-bool.tmp | sed 's/^\t$///g' > laptop-test.bc

restaurant-features:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFVectorFile -Dexec.args="${RESTAURANT_TRAIN} ${EMBEDDIING_FILE} ${embed} restaurant-train.tsv restaurant-train.con"
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppCreateCRFVectorFile -Dexec.args="${RESTAURANT_TEST} ${EMBEDDIING_FILE} ${embed} restaurant-test.tsv restaurant-test.con"
	cat restaurant-train.tsv | ./absa.py -s '\t' > restaurant-train.bin
	cat restaurant-test.tsv | ./absa.py -s '\t' > restaurant-test.bin
	cut -f 2- restaurant-train.bin > restaurant-train-bool.tmp
	cut -f 2- restaurant-test.bin > restaurant-test-bool.tmp
	paste restaurant-train.con restaurant-train-bool.tmp | sed 's/^\t$///g' > restaurant-train.bc
	paste restaurant-test.con restaurant-test-bool.tmp | sed 's/^\t$///g' > restaurant-test.bc

#type=bin(binary) or con(continuous) or bc (binary-continuous)
run-crfsuite:
	#crfsuite learn -m ${dataset}-crfsuite.mdl ${dataset}-train.${type}
	crfsuite learn -a l2sgd -p c2=2.0 -p feature.possible_transitions=1 -p feature.possible_states=1 -m ${dataset}-crfsuite.mdl ${dataset}-train.${type}
	#crfsuite learn -a lbfgs -p c2=1 -p feature.possible_transitions=1 -p feature.possible_states=1 -m ${dataset}-crfsuite.mdl ${dataset}-train.${type}
	crfsuite tag -r -m ${dataset}-crfsuite.mdl ${dataset}-test.${type} > ${dataset}-result.tsv
	paste ${dataset}-test.tsv ${dataset}-result.tsv > ${dataset}-combine.tsv
	cat ${dataset}-combine.tsv | cut -f1,2,4,5 | perl conlleval.pl -d "\t"

###################################################################################################
## Experiments for Cross Validation
###################################################################################################
prepare-folds:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppPrepareDataFolds \
	-Dexec.args="-f cross-validation/${dataset}.tsv -n 10 -d ${dataset}"

# example: make prepare-json dataset=dse fold=0 embed=senna
prepare-json:
	mvn -q compile exec:java -Dexec.mainClass=edu.cuhk.hccl.AppPrepareRnnJson \
	-Dexec.args="-t ${dataset}/train${fold}.tsv -r 0.9 -s ${dataset}/test${fold}.tsv -o ${dataset}-json-${embed}.txt -e ${EMBEDDIING_FILE} -p ${embed}"	
    
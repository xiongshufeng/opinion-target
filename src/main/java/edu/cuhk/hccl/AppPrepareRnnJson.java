package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.xml.bind.JAXBException;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;

import com.google.gson.Gson;

import edu.cuhk.hccl.data.WordAnnotation;
import edu.cuhk.hccl.nlp.TextProcessor;

public class AppPrepareRnnJson {
	
	private static WordAnnotation annotation = null;

	public static void main(String[] args) throws JAXBException, IOException, ParseException {

		System.out.println("[INFO] Processing is started...");

		CommandLineParser parser = new BasicParser();
		Options options = createOptions();
		CommandLine line = parser.parse(options, args);

		// Get parameters
		String trainFile = line.getOptionValue('t');
		String testFile = line.getOptionValue('s');
		String embeddingFile = line.getOptionValue('e');
		String type = line.getOptionValue('p');
		String outFile = line.getOptionValue('o');
		String validFile = line.getOptionValue('v');

		// Read trainFile/testFile and construct trainSet/testSet
		List<List<String>> trainSet = loadDataset(trainFile);
		List<List<String>> testSet = loadDataset(testFile);
		
		// Split trainSet into devSet and valSet
		List<List<String>> devSet = new ArrayList<List<String>>();
		List<List<String>> valSet = new ArrayList<List<String>>();
		
		if (validFile != null) {
			devSet = trainSet;
			valSet = loadDataset(validFile);
		} else {
			float devRatio = Float.parseFloat(line.getOptionValue('r'));
			int devSize = (int) (trainSet.size() * devRatio);
			for (List<String> tmpList : trainSet) {
				double random = Math.random();
				if (random <= devRatio && devSet.size() < devSize)
					devSet.add(tmpList);
				else
					valSet.add(tmpList);
			}
		}
		
		System.out.println("[INFO] Development set size is : " + devSet.size());
		System.out.println("[INFO] Validation set size is : " + valSet.size());
		System.out.println("[INFO] Testing set size is : " + testSet.size());
		
		// Build the vocabulary from the development set
		Map<String, Integer> wordCounter = new TreeMap<String, Integer>();
		for (List<String> sentence : devSet) {
			for (String wordLine : sentence) {
				String word = wordLine.split("\t")[0].toLowerCase();
				
				// Ignore non-word characters
				if (!word.matches("[a-zA-Z]+"))
					continue;
				
				int counter = 1;
				if (wordCounter.containsKey(word)) {
					counter += wordCounter.get(word);
				}
				wordCounter.put(word, counter);
			}
		}
		
		// Create the one-to-one word-index mapping
		Map<String, Integer> vocabulary = new TreeMap<String, Integer>();
		int index = 0;
		for (String key : wordCounter.keySet()) {
			int counter = wordCounter.get(key);
			if (counter == 1) {
				key = Utility.UNKNOWN;
			}

			if (!vocabulary.containsKey(key)) {
				vocabulary.put(key, index);
				index = index + 1;
			}
		}
		
		System.out.println("[INFO] The vocabulary size is : " + vocabulary.keySet().size());
		
		// Load the word embeddings for all words in the vocabulary
		WordEmbedding embedLoader = Utility.loadWordEmbedding(type, embeddingFile);

		float[][] embeddings = new float[vocabulary.keySet().size() + 1][];
		for (String word : vocabulary.keySet()) {
			float[] wordEmbed = embedLoader.getWordEmbedding(word);
			if (wordEmbed == null)
				wordEmbed = embedLoader.getWordEmbedding(Utility.UNKNOWN);

			int wordIdx = vocabulary.get(word);
			embeddings[wordIdx] = wordEmbed;
		}
		
		embeddings[vocabulary.keySet().size()] = embedLoader.getWordEmbedding(Utility.PADDING);
		
		// Create the one-to-one label-index mapping
		annotation = new WordAnnotation(WordAnnotation.createAnnotations("target"));
		Map<String, Integer> label2idx = annotation.createLabelIndexMap();
		
		// Construct the development set, the validation set and the test set
		RnnDataset devSetRnn = createRnnDataset(devSet, vocabulary);
		RnnDataset valSetRnn = createRnnDataset(valSet, vocabulary);
		RnnDataset testSetRnn = createRnnDataset(testSet, vocabulary);
		
		// Output results in JSON
		Gson gson = new Gson();
		
		JsonOutput output = new JsonOutput();
		output.development = devSetRnn;
		output.validate = valSetRnn;
		output.test = testSetRnn;
		output.word2Idx = vocabulary;
		output.label2Idx = label2idx;
		output.embeddings = embeddings;
		
		FileUtils.write(new File(outFile), gson.toJson(output));
		
		System.out.println("[INFO] Result is written to: " + outFile);
		System.out.println("[INFO] Processing is finished!");
	}

	private static List<List<String>> loadDataset(String trainFile) throws IOException {
		List<String> dataLines = FileUtils.readLines(new File(trainFile));
		List<List<String>> trainSet = new ArrayList<List<String>>();
		List<String> record = new ArrayList<String>();
		for (String strLine : dataLines) {
			record.add(strLine);
			if (strLine.isEmpty()) {
				List<String> tmpList = new ArrayList<String>();
				tmpList.addAll(record);
				trainSet.add(tmpList);

				record.clear();
			}
		}
		return trainSet;
	}

	private static RnnDataset createRnnDataset(List<List<String>> dataSet,
			Map<String, Integer> vocabulary) {

		List<int[]> xIndexes = new ArrayList<int[]>();
		List<int[]> yLabels = new ArrayList<int[]>();
		List<int[][]> xFeatures = new ArrayList<int[][]>();
		for (List<String> sentence : dataSet) {

			// The last line is empty
			int numWords = sentence.size() - 1;
			int[] wordsIdx = new int[numWords];
			int[] yLabel = new int[numWords];
			int[][] xFeature = new int[numWords][TextProcessor.FEATURES.size()];

			StringBuffer sentBuffer = new StringBuffer();

			for (int i = 0; i < numWords; i++) {
				String line = sentence.get(i);

				String[] tmpArr = line.split("\t");
				String word = tmpArr[0].toLowerCase().trim();
				String label = tmpArr[2].trim();

				sentBuffer.append(word);
				if (i != numWords - 1)
					sentBuffer.append(" ");

				if (vocabulary.containsKey(word))
					wordsIdx[i] = vocabulary.get(word);	
				else
					wordsIdx[i] = vocabulary.get(Utility.UNKNOWN);
				
				yLabel[i] = annotation.annotationToNumber(label);
			}

			// Fill linguistic features
			Utility.fillFeatures(numWords, xFeature, sentBuffer.toString());

			// Update record
			xIndexes.add(wordsIdx);
			yLabels.add(yLabel);
			xFeatures.add(xFeature);
		}

		RnnDataset rnnSet = new RnnDataset();
		rnnSet.xIndexes = xIndexes;
		rnnSet.yLabels = yLabels;
		rnnSet.xFeatures = xFeatures;

		return rnnSet;
	}
	
	public static Options createOptions() {

		Options options = new Options();

		options.addOption("t", "train", true, "Train file");
		options.addOption("r", "ratio", true, "Split ratio of development and validation");
		options.addOption("s", "test", true, "Test file");
		options.addOption("e", "embedding", true, "Embedding file");
		options.addOption("p", "type", true, "Embedding type");
		options.addOption("l", "label", true, "Label kind: target/restEntity/restAttribute/polarity");
		options.addOption( "o", "output", true, "Output file" );
		options.addOption( "v", "valid", true, "Valid file" );
		
		return options;
	}
}

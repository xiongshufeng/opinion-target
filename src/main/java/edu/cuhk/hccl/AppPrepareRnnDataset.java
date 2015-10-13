package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.TreeMap;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;

import com.google.gson.Gson;

import edu.cuhk.hccl.data.Sentences;
import edu.cuhk.hccl.data.Sentences.Sentence;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms.AspectTerm;
import edu.cuhk.hccl.data.WordAnnotation;
import edu.cuhk.hccl.nlp.TextProcessor;

public class AppPrepareRnnDataset {
	
	private static WordAnnotation annotation = null;
	
	public static void main(String[] args) throws JAXBException, IOException, ParseException {

		System.out.println("[INFO] Processing is started...");

		CommandLineParser parser = new BasicParser();
		Options options = Utility.createOptions();
		CommandLine line = parser.parse(options, args);

		// Get parameters
		String trainFile = line.getOptionValue('t');
		float devRatio = Float.parseFloat(line.getOptionValue('r'));
		String testFile = line.getOptionValue('s');
		String embeddingFile = line.getOptionValue('e');
		String type = line.getOptionValue('p');
		String outFile = line.getOptionValue('o');

		// Read trainFile/testFile and construct trainSet/testSet
		List<SentenceAnnotation> trainSet = createDataset(trainFile);
		List<SentenceAnnotation> testSet = createDataset(testFile);
		
		// Split trainSet into devSet and valSet
		List<SentenceAnnotation> devSet = new ArrayList<SentenceAnnotation>();
		List<SentenceAnnotation> valSet = new ArrayList<SentenceAnnotation>();
		int devSize = (int) (trainSet.size() * devRatio);
		
		for (SentenceAnnotation sentAnno : trainSet){
			double random = Math.random();
			if (random <= devRatio && devSet.size() < devSize) {
				devSet.add(sentAnno);
			} else {
				valSet.add(sentAnno);
			}
		}
		
		// Build the vocabulary from the development set
		Map<String, Integer> wordCounter = new TreeMap<String, Integer>();
		for (SentenceAnnotation sentAnno : devSet) {
			String[] words = sentAnno.sentence.split(Utility.SPACE);
			
			// Store words which are aspect terms
			List<String> devTerms = new ArrayList<String>();
			Iterator<WordAnnotation> iter = sentAnno.annotation.iterator();
			while(iter.hasNext()){
				devTerms.add(iter.next().getWord());
			}
			
			for (String word : words) {
				int counter = 1;
				
				if (devTerms.contains(word)) // Do not mark TARGETS as UNKNOWN
					counter = 2;
				
				if (wordCounter.containsKey(word)) {
					counter += wordCounter.get(word);
				}
				
				wordCounter.put(word, counter);
			}
		}
		
		// Create the one-to-one word-index mapping
		Map<String, Integer> vocabulary = new TreeMap<String, Integer>();
		int index = 0;
		for (String key : wordCounter.keySet()){
			int counter = wordCounter.get(key);
			if (counter == 1){
				key = Utility.UNKNOWN;
			} 
			
			if (!vocabulary.containsKey(key)){
				vocabulary.put(key, index);
				index = index + 1;
			}
		}
		
		// Load the word embeddings for all words in the vocabulary		
		WordEmbedding embedLoader = Utility.loadWordEmbedding(type, embeddingFile);
		
		float[][] embeddings = new float[vocabulary.keySet().size() + 1][];
		for (String word : vocabulary.keySet()){
			float[] wordEmbed = embedLoader.getWordEmbedding(word);
			if (wordEmbed==null)
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

	private static List<SentenceAnnotation> createDataset(String dataFileName)
			throws JAXBException {
		
		JAXBContext context = JAXBContext.newInstance(Sentences.class);
		Unmarshaller unmarshaller = context.createUnmarshaller();

		Sentences trainSents = (Sentences) unmarshaller.unmarshal(new File(dataFileName));
		
		List<SentenceAnnotation> trainSet = new ArrayList<SentenceAnnotation>();
		
		for (Sentence sent : trainSents.getSentence()) {
			
			String text = sent.getText().replaceAll("\\W", Utility.SPACE);
			
			StringBuffer textBuffer = Utility.preprocess(text);
			
			SentenceAnnotation sentAnno = new SentenceAnnotation();
			sentAnno.sentence = textBuffer.toString().trim();
			
			Queue<WordAnnotation> termsQueue = extractAnnotation(sent);
			
			sentAnno.annotation = termsQueue;
			
			trainSet.add(sentAnno);
		}
		
		return trainSet;
	}

	private static Queue<WordAnnotation> extractAnnotation(Sentence sent) {
		AspectTerms terms = sent.getAspectTerms();
		Queue<WordAnnotation> termsQueue = new LinkedList<WordAnnotation>();

		if (terms != null){
			List<AspectTerm> aspects = sent.getAspectTerms().getAspectTerm();
			for(AspectTerm term : aspects){
				String[] words = term.getTerm().split(" ");
				
				Utility.putWordAnnotation(termsQueue, words[0], "TERM", true);

				for (int i = 1; i < words.length; i++){
					Utility.putWordAnnotation(termsQueue, words[i], "TERM", false);
				}
			}
		}
		return termsQueue;
	}

	private static RnnDataset createRnnDataset(List<SentenceAnnotation> dataSet,
			Map<String, Integer> vocabulary) {
		
		List<int[]> xIndexes = new ArrayList<int[]>();
		List<int[]> yLabels = new ArrayList<int[]>();
		List<int[][]> xFeatures = new ArrayList<int[][]>();
		
		for (SentenceAnnotation sentAnno : dataSet){
			String[] words = sentAnno.sentence.split(Utility.SPACE);
			Queue<WordAnnotation> annoQueue = sentAnno.annotation;
			
			int[] wordsIdx = new int[words.length];
			int[] yLabel = new int[words.length];
			int[][] xFeature = new int[words.length][TextProcessor.FEATURES.size()];
			
			for (int i = 0; i < words.length; i++){

				if (vocabulary.containsKey(words[i]))
					wordsIdx[i] = vocabulary.get(words[i]);	
				else
					wordsIdx[i] = vocabulary.get(Utility.UNKNOWN);
				
				WordAnnotation peek = annoQueue.peek();
				if (peek != null && words[i].equalsIgnoreCase(peek.getWord())) {
					yLabel[i] = annotation.annotationToNumber(peek.getAnnotation());
					annoQueue.remove();
				} else {
					yLabel[i] = annotation.annotationToNumber(WordAnnotation.OTHER);
				}
			}

			// Fill linguistic features
			Utility.fillFeatures(words.length, xFeature, sentAnno.sentence);
			
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
}

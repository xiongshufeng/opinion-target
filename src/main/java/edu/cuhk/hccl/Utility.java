package edu.cuhk.hccl;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import opennlp.tools.chunker.ChunkerME;

import org.apache.commons.cli.Options;

import edu.cuhk.hccl.data.WordAnnotation;
import edu.cuhk.hccl.nlp.Parser;
import edu.cuhk.hccl.nlp.TextProcessor;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class Utility {
	
	public static final String SPACE = " ";
	public static final String UNKNOWN = "UNKNOWN";
	public static final String PADDING = "PADDING";

	public static StanfordCoreNLP tagger = Parser.createTagger();
	public static ChunkerME chunker = TextProcessor.loadChunker();
	
	public static StringBuffer preprocess(String text) {
		
		StringBuffer textBuffer = new StringBuffer();
		String[] words = text.toLowerCase().split(SPACE);
		for (String word : words) {
			
			if (word.isEmpty())
				continue;
			
			if (word.matches("\\d+")) {
				for (int i = 0; i < word.length(); i++) {
					textBuffer.append("DIGIT");
				}
			} else{
				textBuffer.append(word);
			}
			
			textBuffer.append(SPACE);
		}
		return textBuffer;
	}

	public static void putWordAnnotation(Queue<WordAnnotation> termsQueue,
			String word, String label, boolean isBegin) {
	
		String termAnnotation = WordAnnotation.labelToAnnotation(label, isBegin);
		termsQueue.add(new WordAnnotation(word, termAnnotation ));
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
		
		return options;
	}
	
	public static WordEmbedding loadWordEmbedding(String type, String embeddingFile)
			throws IOException {

		WordEmbedding embedLoader = null;

		if (type.equalsIgnoreCase("Google")) {
			embedLoader = new GoogleWordEmbedding(embeddingFile);
		} else if (type.equalsIgnoreCase("Senna")) {
			embedLoader = new SennaWordEmbedding(embeddingFile);
		} else if (type.equalsIgnoreCase("Amazon")) {
			embedLoader = new AmazonWordEmbedding(embeddingFile);
		} else {
			System.out.println("[ERROR] Unsupported embedding type!");
			System.exit(-1);
		}
		embedLoader.loadWordVectors();

		return embedLoader;
	}

	public static void fillFeatures(int numWords, int[][] xFeature,
			String sentStr) {
		
		Annotation tagAnno = tagger.process(sentStr);
		Map<String, String> wordTagMap = new HashMap<String, String>();
		Map<String, String> wordChunkMap = new HashMap<String, String>();

		List<CoreMap> tmpMapList = tagAnno.get(CoreAnnotations.SentencesAnnotation.class);
		
		if (tmpMapList != null && tmpMapList.size() == 1) {
			CoreMap sentMap = tmpMapList.get(0);

			List<CoreLabel> labels = sentMap.get(TokensAnnotation.class);
			String[] chunkWords = new String[labels.size()];
			String[] chunkPos = new String[labels.size()];
			for (int i = 0; i < labels.size(); i++) {
				CoreLabel label = labels.get(i);
				chunkWords[i] = label.originalText();
				chunkPos[i] = label.tag();
			}

			String[] chunkTags = chunker.chunk(chunkWords, chunkPos);
			for (int i = 0; i < chunkTags.length; i++) {
				wordTagMap.put(chunkWords[i], chunkPos[i]);
				wordChunkMap.put(chunkWords[i], chunkTags[i]);
			}

			// Fill POS and Chunk Features
			String[] tmpWords = sentStr.split(" ");
			for (int i = 0; i < numWords; i++) {
				String tag = wordTagMap.get(tmpWords[i]);
				if (TextProcessor.JJ_TAGS.contains(tag))
					xFeature[i][TextProcessor.FEATURES.indexOf("JJ")] = 1;
				else if (TextProcessor.NN_TAGS.contains(tag))
					xFeature[i][TextProcessor.FEATURES.indexOf("NN")] = 1;
				else if (TextProcessor.RB_TAGS.contains(tag))
					xFeature[i][TextProcessor.FEATURES.indexOf("RB")] = 1;
				else if (TextProcessor.VB_TAGS.contains(tag))
					xFeature[i][TextProcessor.FEATURES.indexOf("VB")] = 1;

				String chunk = wordChunkMap.get(tmpWords[i]);
				int chunkIndex = TextProcessor.FEATURES.indexOf(chunk);
				if (chunkIndex != -1)
					xFeature[i][chunkIndex] = 1;
			}
		}
	}
	
}

class SentenceAnnotation{
	
	String sentence;
	Queue<WordAnnotation> annotation;
}

class JsonOutput{

	RnnDataset development;
	RnnDataset validate;
	RnnDataset test;
	
	Map<String, Integer> word2Idx;
	Map<String, Integer> label2Idx;
	float[][] embeddings;
}

class RnnDataset{
	
	// Store word indexes and labels
	List<int[]> xIndexes;
	List<int[]> yLabels;
	
	// Store linguistic features
	List<int[][]> xFeatures;
}

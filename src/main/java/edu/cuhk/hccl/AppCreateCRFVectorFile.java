package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;

import org.apache.commons.io.FileUtils;

import edu.cuhk.hccl.data.Sentences;
import edu.cuhk.hccl.data.Sentences.Sentence;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms.AspectTerm;
import edu.cuhk.hccl.data.WordAnnotation;
import edu.cuhk.hccl.nlp.Parser;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class AppCreateCRFVectorFile {
	
	public static final String SEPERATOR = "\t";
	public static final String SPACE = " ";
	
	private static WordEmbedding embedLoader = null;

	public static void main(String[] args) throws IOException, JAXBException {

		System.out.println("Processing is started...");

		String dataFile = args[0];
		String embeddingFile = args[1];
		String type = args[2];
		String crfFile = args[3];
		String vecFile = args[4];
		
		// Load the word embeddings for all words in the vocabulary
		embedLoader = Utility.loadWordEmbedding(type, embeddingFile);
		
		// Create files
		createCRFfile(dataFile, crfFile, vecFile);

		System.out.println("Processing is finished!");
	}

	public static void createCRFfile(String dataFile, String crfFile, String vecFile)
			throws IOException, JAXBException {

		JAXBContext context = JAXBContext.newInstance(Sentences.class);
		Unmarshaller unmarshaller = context.createUnmarshaller();
		Sentences sents = (Sentences) unmarshaller.unmarshal(new File(dataFile));
		
		List<String> lines = new ArrayList<String>();
		List<String> vecLines = new ArrayList<String>();
		StanfordCoreNLP tagger = Parser.createTagger();
		
		for (Sentence sent : sents.getSentence()) {
			
			AspectTerms terms = sent.getAspectTerms();
			Queue<WordAnnotation> termsQueue = new LinkedList<WordAnnotation>();
			
			if (terms != null){
				List<AspectTerm> aspects = sent.getAspectTerms().getAspectTerm();
				for(AspectTerm term : aspects){
					String[] words = term.getTerm().split(" ");
					termsQueue.add(new WordAnnotation(words[0],WordAnnotation.labelToAnnotation("TERM", true)));
					
					for (int i = 1; i < words.length; i++){
						termsQueue.add(new WordAnnotation(words[i], WordAnnotation.labelToAnnotation("TERM", false)));
					}
				}
			}
			
			Annotation annotation = tagger.process(sent.getText());
			for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
				List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
				for (CoreLabel label : labels){
					WordAnnotation peek = termsQueue.peek();
					String originalText = label.originalText();
					
					if (!originalText.isEmpty()) {
						
						String wordVec = getWordEmbedding(originalText);
						
						if (peek != null && originalText.equalsIgnoreCase(peek.getWord())) {
							lines.add(originalText + SEPERATOR + label.tag() + SEPERATOR
									+ peek.getAnnotation());
							vecLines.add(peek.getAnnotation() + SEPERATOR + wordVec);
							termsQueue.remove();
						} else {
							lines.add(originalText + SEPERATOR + label.tag() + SEPERATOR
									+ WordAnnotation.OTHER);
							vecLines.add(WordAnnotation.OTHER + SEPERATOR + wordVec);
						}
					}
				}
			}

			lines.add("");
			vecLines.add("");
		}

		FileUtils.writeLines(new File(crfFile), lines, false);
		FileUtils.writeLines(new File(vecFile), vecLines, false);
	}

	private static String getWordEmbedding(String word) {
		
		float[] wordEmbed = embedLoader.getWordEmbedding(word);
		if (wordEmbed==null)
			wordEmbed = embedLoader.getWordEmbedding(Utility.UNKNOWN);
		
		StringBuffer features = new StringBuffer();
		
		for (int i = 0; i < wordEmbed.length; i++){
			String feature = String.format("v[%d]=v%d:%.6f%s", i, i, wordEmbed[i], SEPERATOR);
			features.append(feature);
		}
		
		String wordVec = features.toString().trim();
		return wordVec;
	}
	
}